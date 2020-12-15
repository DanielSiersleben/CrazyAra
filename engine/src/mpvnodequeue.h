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

    bool batch_ready;

    size_t totalEvals;

    mutex* mtx;
    unique_ptr<float[]> inputPlanes;

    // store nodes here while inputPlanes is processed by GPU
    unique_ptr<float[]> inputBuffer;
    unique_ptr<Node*[]> queueBuffer;
    unique_ptr<SideToMove[]> sideToMoveBuffer;
    unique_ptr<Trajectory[]> trajectoriesBuffer;

    MPVNodeQueue(size_t batchSize){
        queue = make_unique<Node*[]>(batchSize);
        sideToMove = make_unique<SideToMove[]>(batchSize);
        trajectories = make_unique<Trajectory[]>(batchSize);
        this->mtx = new mutex();
        batchIdx = new atomic_int(0);
        this->batchSize = batchSize;

        batch_ready = false;

        totalEvals = 0;

        this->inputPlanes = make_unique<float[]>(batchSize * StateConstants::NB_VALUES_TOTAL());

        this->inputBuffer = make_unique<float[]>(batchSize * StateConstants::NB_VALUES_TOTAL());
        queueBuffer = make_unique<Node*[]>(batchSize);
        sideToMoveBuffer = make_unique<SideToMove[]>(batchSize);
        trajectoriesBuffer = make_unique<Trajectory[]>(batchSize);
    }

    void mark_batch_completed(){
        batch_ready = false;
        totalEvals += batchSize;

    }

    void setInputPlanesAndBuffer(float* inputPlanes, float* inputBuffer){
        this->inputPlanes.reset(inputPlanes);
        this->inputBuffer.reset(inputBuffer);
    }

    void clear(){

        queue.reset(new Node*[batchSize]);
        sideToMove.reset(new SideToMove[batchSize]);
        trajectories.reset(new Trajectory[batchSize]);
        queueBuffer.reset(new Node*[batchSize]);
        sideToMoveBuffer.reset(new SideToMove[batchSize]);
        trajectoriesBuffer.reset(new Trajectory[batchSize]);


        batchIdx->store(0);
        batch_ready = false;

        totalEvals = 0;
    }

    int fetch_and_increase_Index(){
       int tmp = batchIdx->fetch_add(1);

       while (tmp >= batchSize) {
           // wait till Buffer is swapped
           tmp = batchIdx->fetch_add(1);
       }

       return tmp;
    }

    void insert(Node* node, SideToMove side, Trajectory trajectory, int index){
        trajectory.pop_back();

        this->queue[index] = node;
        this->sideToMove[index] = side;
        this->trajectories[index] = trajectory;

        if(index == batchSize-1){
            // swap pointer of Buffer and Planes
            swapBuffer();
        }
    }

    void swapBuffer(){
        // make sure Buffer is already completed
        assert(batch_ready == false);

        queue.swap(queueBuffer);
        sideToMove.swap(sideToMoveBuffer);
        trajectories.swap(trajectoriesBuffer);
        inputPlanes.swap(inputBuffer);
        batchIdx->store(0);
        batch_ready = true;
    }

    // returns the completed Buffer
    float* getInputBuffer(){
        return inputBuffer.get();
    }

    float* getInputPlanes(){
        return inputPlanes.get();
    }

    size_t getLargeNetEvals(){
        return totalEvals;
    }
};

#endif // MPVNODEQUEUE_H
