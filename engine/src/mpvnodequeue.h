#ifndef MPVNODEQUEUE_H
#define MPVNODEQUEUE_H
#ifdef MPV_MCTS

#include "node.h"
#include "atomic"

struct MPVNodeQueue{
    unique_ptr<Node*[]> queue;
    unique_ptr<SideToMove[]> sideToMove;
    unique_ptr<Trajectory[]> trajectories;
    int batchSize;
    atomic_size_t* batchIdx;
    mutex* mtx;
    unique_ptr<float[]> inputPlanes;

    // store nodes here while inputPlanes is processed by GPU
    unique_ptr<float[]> inputBuffer;
    bool dataBuffered;

    MPVNodeQueue(size_t batchSize, mutex* mtx, atomic_size_t* idx){
        queue = make_unique<Node*[]>(2*batchSize);
        sideToMove = make_unique<SideToMove[]>(2*batchSize);
        trajectories = make_unique<Trajectory[]>(2*batchSize);
        this->mtx = mtx;
        batchIdx = idx;
        this->batchSize = batchSize;

        inputBuffer = make_unique<float[]>(batchSize * StateConstants::NB_VALUES_TOTAL());
        dataBuffered = false;
    }


    MPVNodeQueue(){}

    void setInputPlanes(float* iP){
        this->inputPlanes = unique_ptr<float[]>(iP);
    }
    void clear(){
        mtx->lock();
        size_t currIdx = batchIdx->load();
        if(currIdx < batchSize){
            for(int idx = 0; idx < currIdx; ++idx){
                    queue[idx]->disable_node_is_enqueued();
            }
            batchIdx->store(0);
        }
        else{
            if(dataBuffered){
                currIdx -= batchSize;
                for(auto newIdx = 1; newIdx <= currIdx; ++newIdx){
                    for(auto i = 0; i < StateConstants::NB_VALUES_TOTAL(); ++i){
                        inputPlanes[newIdx*i] = inputBuffer[newIdx*i];
                    }
                    queue[newIdx-1] = queue[batchSize+newIdx-1];
                    sideToMove[newIdx-1] = sideToMove[batchSize+newIdx-1];
                    trajectories[newIdx-1] = trajectories[batchSize+newIdx-1];
                }
                batchIdx->store(currIdx);
                dataBuffered = false;
            }
            else{
                batchIdx->store(0);
            }

        }

        mtx->unlock();
    }

    /*void emplace_back(Node* node, SideToMove side, Trajectory trajectory){
        this->queue.emplace_back(node);
        this->sideToMove.emplace_back(side);
        this->trajectories.emplace_back(trajectory);
        batchIdx++;
    }*/

    int fetch_and_increase_Index(){
       return batchIdx->fetch_add(1);
    }

    void insert(Node* node, SideToMove side, Trajectory trajectory, int index){
        this->queue[index] = node;
        this->sideToMove[index] = side;
        this->trajectories[index] = trajectory;
    }

    float* getInputPlanes(){
        return inputPlanes.get();
        if(batchIdx->load() >= batchSize){
            dataBuffered = true;
            return inputBuffer.get();
        }
        else return inputPlanes.get();
    }
};

#endif
#endif // MPVNODEQUEUE_H
