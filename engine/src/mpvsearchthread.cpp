#include "mpvsearchthread.h"
#ifdef MPV_MCTS

MPVSearchThread::MPVSearchThread(NeuralNetAPI* netBatch, SearchSettings* searchSettings, MapWithMutex* mapWithMutex, MPVNodeQueue *nodeQueue):
    SearchThread(netBatch, searchSettings, mapWithMutex, nodeQueue)
{
     nodeQueue->inputPlanes = inputPlanes;
}

void MPVSearchThread::reset_stats()
{
    tbHits = 0;
    depthMax = 0;
    depthSum = 0;
    nodeQueue->clear();
    // not sure if necessary
    nodeQueue->inputPlanes = inputPlanes;
}

void MPVSearchThread::create_mini_batch()
{
    if(nodeQueue->batchIdx > searchSettings->batchSize){
        cout << "error" << endl;
    }
   if(nodeQueue->batchIdx == searchSettings->batchSize){
       for(size_t i = 0; i < searchSettings->batchSize; ++i){
           newNodes->add_element(nodeQueue->queue[i]);
           newNodeSideToMove->add_element(nodeQueue->sideToMove[i]);
           newTrajectories.emplace_back(nodeQueue->trajectories[i]);
       }
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
    create_mini_batch();
    if (newNodes->size() != 0) {
        net->predict(inputPlanes, valueOutputs, probOutputs);
        set_nn_results_to_child_nodes();
        nodeQueue->clear();
        backup_value_outputs();

        newNodeSideToMove->reset_idx();
    }

}

void MPVSearchThread::backup_mpvnet_values(FixedVector<Node*>* nodes, vector<Trajectory>& trajectories)
{
    for(size_t idx = 0; idx < nodes->size(); ++idx){
        const Node* node = nodes->get_element(idx);
        backup_mpv_value(node->get_value(), searchSettings->virtualLoss, trajectories[idx], searchSettings->largeNetEvalThreshold);
    }
    nodes->reset_idx();
    trajectories.clear();
}

void MPVSearchThread::backup_value_outputs()
{
    backup_mpvnet_values(newNodes.get(), newTrajectories);
}

void fill_mpvnn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, Node *node, size_t& tbHits, SideToMove sideToMove, const SearchSettings* searchSettings)
{
    node->set_probabilities_for_moves(get_policy_data_batch(batchIdx, probOutputs, isPolicyMap), sideToMove);
    node_post_process_policy(node, searchSettings->nodePolicyTemperature, isPolicyMap, searchSettings);
    node_assign_value(node, valueOutputs, tbHits, batchIdx);
    node->enable_has_large_nn_results();
}
#endif
