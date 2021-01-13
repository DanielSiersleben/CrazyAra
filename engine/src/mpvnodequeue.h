#ifndef MPVNODEQUEUE_H
#define MPVNODEQUEUE_H

#include "node.h"
#include "atomic"

class MPVSearchThread;


class MPVNodeQueue{
public:
    atomic_bool* mpvThread_active;

private:
    unique_ptr<Node*[]> queue;
    unique_ptr<SideToMove[]> sideToMove;
    unique_ptr<Trajectory[]> trajectories;

    const size_t batchSize;

    atomic_int* batchIdx;
    atomic_int* completedIdx;

    mutex* mtx;

    bool batch_ready;

    size_t totalEvals;

    float* inputPlanes;

    // store nodes here while inputPlanes is processed by GPU
    float* inputBuffer;
    unique_ptr<Node*[]> queueBuffer;
    unique_ptr<SideToMove[]> sideToMoveBuffer;
    unique_ptr<Trajectory[]> trajectoriesBuffer;

public:
    MPVNodeQueue(size_t batchSize):
        batchSize(batchSize)
    {
        queue = make_unique<Node*[]>(batchSize);
        sideToMove = make_unique<SideToMove[]>(batchSize);
        trajectories = make_unique<Trajectory[]>(batchSize);

        batchIdx = new atomic_int(0);
        completedIdx = new atomic_int(0);
        mtx = new mutex();

        batch_ready = false;

        mpvThread_active = new atomic_bool(false);

        totalEvals = 0;

        queueBuffer = make_unique<Node*[]>(batchSize);
        sideToMoveBuffer = make_unique<SideToMove[]>(batchSize);
        trajectoriesBuffer = make_unique<Trajectory[]>(batchSize);
    }
    ~MPVNodeQueue(){
        delete mtx;
        delete batchIdx;
        delete completedIdx;
    }

    Node** getQueue(){
        return queueBuffer.get();
    }

    size_t getIndex(){
        return batchIdx->load();
    }

    size_t getCompletedIndex(){
        return completedIdx->load();
    }

    SideToMove* getSideToMove(){
        return sideToMoveBuffer.get();
    }

    Trajectory* getTrajectories(){
        return trajectoriesBuffer.get();
    }

    bool batch_is_ready(){
        return batch_ready;
    }

    void mark_batch_completed(){
        batch_ready = false;
        totalEvals += batchSize;

    }

    void setInputPlanesAndBuffer(float* inputPlanes, float* inputBuffer){
        this->inputPlanes = (inputPlanes);
        this->inputBuffer = (inputBuffer);
    }

    void clear(){
        mtx->lock();
        /*
        queue.reset(new Node*[batchSize]);
        sideToMove.reset(new SideToMove[batchSize]);
        trajectories.reset(new Trajectory[batchSize]);
        queueBuffer.reset(new Node*[batchSize]);
        sideToMoveBuffer.reset(new SideToMove[batchSize]);
        trajectoriesBuffer.reset(new Trajectory[batchSize]);*/

        batchIdx->store(0);
        completedIdx->store(0);
        batch_ready = false;

        totalEvals = 0;
        mtx->unlock();
    }

    // important when ReuseTree is used
    void mark_nodes_as_dequeued(){
        int lastIdx = completedIdx->load();
        Node** curr = &queue[0];
        Node** last = &queue[lastIdx];

        while(curr != last){
            (*curr)->disable_node_is_enqueued();
            curr++;
        }
        if(batch_ready){
            curr = &queue[0];
            last = &queue[batchSize-1];
            while(curr != last){
                (*curr)->disable_node_is_enqueued();
                curr++;
            }
        }
    }

    int fetch_and_increase_Index(){
       int tmp = batchIdx->fetch_add(1);

       while (tmp >= batchSize){
           //wait till all insertion completed
           while(completedIdx->load() < batchSize && batchIdx->load() >= batchSize){
               if(!mpvThread_active->load()){
                   break;
               }
           }

           // swap pointer of Buffer and Planes
           if(mtx->try_lock()){
               if(batchIdx->load() >= batchSize){
                   swapBuffer();
               }
               mtx->unlock();
           }
           tmp = batchIdx->fetch_add(1);       
       }


       return tmp;
    }

    void insert(Node* node, SideToMove side, Trajectory trajectory, int index){
        assert(index < batchSize);
        this->queue[index] = node;
        this->sideToMove[index] = side;
        this->trajectories[index] = trajectory;

        completedIdx->fetch_add(1);
    }

    void swapBuffer(){
        assert(batchIdx->load() >= batchSize);
        assert(completedIdx->load() >= batchSize);
        // make sure Buffer is already completed
        while(this->batch_ready){
            if(!mpvThread_active->load()) break;
        }

        float* tmp = inputPlanes;
        inputPlanes = inputBuffer;
        inputBuffer = tmp;

        queue.swap(queueBuffer);
        sideToMove.swap(sideToMoveBuffer);
        trajectories.swap(trajectoriesBuffer);

        batch_ready = true;

        completedIdx->store(0);
        batchIdx->store(0);
    }

    float* getInputBuffer(){
        return inputBuffer;
    }

    float* getInputPlanes(){
        return inputPlanes;
    }

    size_t getLargeNetEvals(){
        return totalEvals;
    }

};

#endif // MPVNODEQUEUE_H
