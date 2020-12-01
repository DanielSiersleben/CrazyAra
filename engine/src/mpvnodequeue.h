#ifndef MPVNODEQUEUE_H
#define MPVNODEQUEUE_H
#ifdef MPV_MCTS

#include "node.h"
#include "atomic"

struct MPVNodeQueue{
    vector<Node*> queue;
    vector<SideToMove> sideToMove;
    int batchIdx;
    mutex* mtx;
    float* inputPlanes;

    MPVNodeQueue(size_t batchSize, mutex* mtx){
        queue = vector<Node*>();
        sideToMove = vector<SideToMove>();
        this->mtx = mtx;
        batchIdx = 0;
    }
    MPVNodeQueue(){}

    void setInputPlanes(float* iP){
        this->inputPlanes = iP;
    }
    void clear(){
        mtx->lock();
        if(!queue.empty()){
            for(auto node : queue){
                node->disable_node_is_enqueued();
            }
        }
        queue.clear();
        sideToMove.clear();
        batchIdx = 0;
        mtx->unlock();
    }

    void emplace_back(Node* node, SideToMove side){
        this->queue.emplace_back(node);
        this->sideToMove.emplace_back(side);
        batchIdx++;
    }
    ~MPVNodeQueue(){}
};

#endif
#endif // MPVNODEQUEUE_H
