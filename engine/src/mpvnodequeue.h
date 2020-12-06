#ifndef MPVNODEQUEUE_H
#define MPVNODEQUEUE_H
#ifdef MPV_MCTS

#include "node.h"
#include "atomic"

struct MPVNodeQueue{
    Node** queue;
    SideToMove* sideToMove;
    Trajectory* trajectories;
    int batchSize;
    int batchIdx;
    mutex* mtx;
    float* inputPlanes;

    // store nodes here while inputPlanes is processed by GPU
    float* inputBuffer;
    bool dataBuffered;

    MPVNodeQueue(size_t batchSize, mutex* mtx){
        queue = new Node*[2*batchSize];
        sideToMove = new SideToMove[2*batchSize];
        trajectories = new Trajectory[2*batchSize];
        this->mtx = mtx;
        batchIdx = 0;
        this->batchSize = batchSize;

        inputBuffer = new float[batchSize * StateConstants::NB_VALUES_TOTAL()];
        std::fill(inputBuffer, inputBuffer + batchSize * StateConstants::NB_VALUES_TOTAL(), 0);
        dataBuffered = false;
    }


    MPVNodeQueue(){}

    /*~MPVNodeQueue(){
        delete [] queue;
        delete [] sideToMove;
        delete [] trajectories;
        delete [] inputBuffer;
    }*/

    void setInputPlanes(float* iP){
        this->inputPlanes = iP;
    }
    void clear(){
        mtx->lock();
        if(batchIdx < batchSize){
            for(int idx = 0; idx < batchIdx; ++idx){
                    queue[idx]->disable_node_is_enqueued();
            }
            batchIdx = 0;
        }
        else{
            if(dataBuffered){
                batchIdx -= batchSize;
                for(auto newIdx = 1; newIdx <= batchIdx; ++newIdx){
                    for(auto i = 0; i < StateConstants::NB_VALUES_TOTAL(); ++i){
                        inputPlanes[newIdx*i] = inputBuffer[newIdx*i];
                    }
                    queue[newIdx-1] = queue[batchSize+newIdx-1];
                    sideToMove[newIdx-1] = sideToMove[batchSize+newIdx-1];
                    trajectories[newIdx-1] = trajectories[batchSize+newIdx-1];
                }
                dataBuffered = false;
            }
            else{
                batchIdx = 0;
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
       mtx->lock();
       int old = batchIdx;
       batchIdx++;
       mtx->unlock();
       return old;
    }

    void insert(Node* node, SideToMove side, Trajectory trajectory, int index){
        this->queue[index] = node;
        this->sideToMove[index] = side;
        this->trajectories[index] = trajectory;
    }

    float* getInputPlanes(){
        return inputPlanes;
        if(batchIdx >= batchSize){
            dataBuffered = true;
            return inputBuffer;
        }
        else return inputPlanes;
    }
};

#endif
#endif // MPVNODEQUEUE_H
