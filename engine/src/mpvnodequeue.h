#ifndef MPVNODEQUEUE_H
#define MPVNODEQUEUE_H

#include "node.h"
#include "atomic"

struct MPVNodeQueue{
    unique_ptr<Node*[]> queue;
    unique_ptr<SideToMove[]> sideToMove;
    unique_ptr<Trajectory[]> trajectories;
    int batchSize;
    atomic_int* batchIdx;
    mutex* mtx;
    unique_ptr<float[]> inputPlanes;

    // store nodes here while inputPlanes is processed by GPU
    unique_ptr<float[]> inputBuffer;
    unique_ptr<Node*[]> queueBuffer;
    unique_ptr<SideToMove[]> sideToMoveBuffer;
    unique_ptr<Trajectory[]> trajectoriesBuffer;
    bool dataBuffered;

    MPVNodeQueue(size_t batchSize, mutex* mtx, atomic_int* idx){
        queue = make_unique<Node*[]>(batchSize);
        sideToMove = make_unique<SideToMove[]>(batchSize);
        trajectories = make_unique<Trajectory[]>(batchSize);
        this->mtx = mtx;
        batchIdx = idx;
        this->batchSize = batchSize;

        this->inputBuffer = make_unique<float[]>(batchSize * StateConstants::NB_VALUES_TOTAL());
        queueBuffer = make_unique<Node*[]>(batchSize);
        sideToMoveBuffer = make_unique<SideToMove[]>(batchSize);
        trajectoriesBuffer = make_unique<Trajectory[]>(batchSize);
        dataBuffered = false;
    }



    void setInputPlanes(float* iP){
        this->inputPlanes.reset(iP);
    }

    void clear(){
        int currIdx = batchIdx->load();

        queue.reset(new Node*[batchSize]);
        sideToMove.reset(new SideToMove[batchSize]);
        trajectories.reset(new Trajectory[batchSize]);
        queueBuffer.reset(new Node*[batchSize]);
        sideToMoveBuffer.reset(new SideToMove[batchSize]);
        trajectoriesBuffer.reset(new Trajectory[batchSize]);
        inputBuffer.reset(new float[batchSize * StateConstants::NB_VALUES_TOTAL()]);

        batchIdx->store(0);
        dataBuffered =  false;
    }

    void resetIdx(){
        batchIdx->store(0);

        /* Buffering Currently Disabled
        int currIdx = batchIdx->load();
        if(currIdx <= batchSize){
            for(int idx = 0; idx < currIdx; ++idx){
                    queue[idx]->disable_node_is_enqueued();
            }
            batchIdx->store(0);
        }
        else{
            assert(dataBuffered);
            currIdx -= batchSize;
            mtx->lock();
            batchIdx->store(currIdx);
            std::copy(inputBuffer.get(), inputBuffer.get() + (currIdx-1)*StateConstants::NB_VALUES_TOTAL(), inputPlanes.get());
            swap(queue, queueBuffer);
            swap(sideToMove, sideToMoveBuffer);
            swap(trajectories, trajectoriesBuffer);

            dataBuffered = false;
            mtx->unlock();

        }*/
    }

    int fetch_and_increase_Index(){
       return batchIdx->fetch_add(1);
    }

    void insert(Node* node, SideToMove side, Trajectory trajectory, int index){
        trajectory.pop_back();
        if(index < batchSize){
            this->queue[index] = node;
            this->sideToMove[index] = side;
            this->trajectories[index] = trajectory;
        }
        else{
            this->queueBuffer[index % batchSize] = node;
            this->sideToMoveBuffer[index % batchSize] = side;
            this->trajectoriesBuffer[index % batchSize] = trajectory;
            mtx->unlock();
        }
    }

    float* getInputPlanes(int idx){
        return inputPlanes.get();

        //currently disabled Buffering
        if(idx >= batchSize){
            dataBuffered = true;
            cout << "buffering" << endl;
            // make sure that buffer isnt swapped while not finished
            return inputBuffer.get();
        }
        else{
            return inputPlanes.get();
        }
    }
};

#endif // MPVNODEQUEUE_H
