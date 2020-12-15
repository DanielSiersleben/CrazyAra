#include "mpvsearchthread.h"
#include <thread>


#ifdef MPV_MCTS

MPVSearchThread::MPVSearchThread(NeuralNetAPI* netBatch, SearchSettings* searchSettings, MapWithMutex* mapWithMutex, MPVNodeQueue *nodeQueue):
    SearchThread(netBatch, searchSettings, mapWithMutex, nodeQueue, true)
{
     newNodes = make_unique<FixedVector<Node*>>(searchSettings->largeNetBatchSize);
     newNodeSideToMove = make_unique<FixedVector<SideToMove>>(searchSettings->largeNetBatchSize);

     nodeQueue->setInputPlanesAndBuffer(inputPlanes, inputBuffer);
}

void MPVSearchThread::reset_stats()
{
    tbHits = 0;
    depthMax = 0;
    depthSum = 0;
    nodeQueue->clear();
    this->workerThreads = new thread*[searchSettings->largeNetBackpropThreads];

}

void MPVSearchThread::create_mpv_mini_batch()
{
       for(auto i = 0; i < nodeQueue->batchSize; ++i){
           newNodes->add_element(nodeQueue->queueBuffer[i]);
           newNodeSideToMove->add_element(nodeQueue->sideToMoveBuffer[i]);
           newTrajectories.emplace_back(nodeQueue->trajectoriesBuffer[i]);
       }
}

void MPVSearchThread::set_nn_results_to_child_nodes()
{
    size_t batchIdx = 0;
    for (auto node: *newNodes) {
        if (!node->is_terminal()) {
            fill_mpvnn_results(batchIdx, net->is_policy_map(), valueOutputs, probOutputs, node, tbHits, newNodeSideToMove->get_element(batchIdx), searchSettings);
        }
        ++batchIdx;
    }
}

void MPVSearchThread::thread_iteration()
{
    if(nodeQueue->batch_ready){
        create_mpv_mini_batch();
    }
    if (newNodes->size() != 0) {
        net->predict(nodeQueue->getInputBuffer(), valueOutputs, probOutputs);
        set_nn_results_to_child_nodes();
        nodeQueue->mark_batch_completed();
        backup_value_outputs();
        newNodeSideToMove->reset_idx();
    }
}

void backup_mpvnet_values(FixedVector<Node*>* nodes, vector<Trajectory>* trajectories, atomic_int* idx, SearchSettings* searchSettings)
{
    int i;
    while((i = idx->fetch_add(1)) < nodes->size()){
        const Node* node = nodes->get_element(i);
        backup_mpv_value(node->get_value(), trajectories->operator[](i), searchSettings->largeNetEvalThreshold);
    }

    /*for(size_t idx = 0; idx < nodes->size(); ++idx){
        const Node* node = nodes->get_element(idx);
        backup_mpv_value(node->get_value(), (*trajectories)[idx], searchSettings->largeNetEvalThreshold);
    }*/
}

void MPVSearchThread::backup_value_outputs()
{
    if(searchSettings->largeNetValueBackprop){
        atomic_int idx = 0;
        for(auto i = 0; i < searchSettings->largeNetBackpropThreads; ++i){
            workerThreads[i] = new thread(backup_mpvnet_values, newNodes.get(), &newTrajectories, &idx, searchSettings);
        }

        for(auto i = 0; i < searchSettings->largeNetBackpropThreads; ++i){
            workerThreads[i]->join();
        }
    }

    newNodes->reset_idx();
    newTrajectories.clear();
}

void MPVSearchThread::set_is_running(bool value)
{
    isRunning = value;
    if(value == false){
        deleteWorkerThreads();
    }
}

void MPVSearchThread::deleteWorkerThreads(){
        delete[] workerThreads;
}


void fill_mpvnn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, Node *node, size_t& tbHits, SideToMove sideToMove, const SearchSettings* searchSettings)
{
    node->set_probabilities_for_moves(get_policy_data_batch(batchIdx, probOutputs, isPolicyMap), sideToMove);
    node_post_process_policy(node, searchSettings->nodePolicyTemperature, isPolicyMap, searchSettings);
    node_assign_value(node, valueOutputs, tbHits, batchIdx);
    node->enable_has_large_nn_results();
}

#endif
