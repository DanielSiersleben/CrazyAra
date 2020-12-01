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
}

void MPVSearchThread::create_mini_batch()
{
   if(nodeQueue->batchIdx == searchSettings->batchSize){
       for(size_t i = 0; i < searchSettings->batchSize; ++i){
           newNodes->add_element(nodeQueue->queue[i]);
           newNodeSideToMove->add_element(nodeQueue->sideToMove[i]);
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
    }
    newNodes->reset_idx();
    newNodeSideToMove->reset_idx();

}

void fill_mpvnn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, Node *node, size_t& tbHits, SideToMove sideToMove, const SearchSettings* searchSettings)
{
    node->set_probabilities_for_moves(get_policy_data_batch(batchIdx, probOutputs, isPolicyMap), sideToMove);
    node_post_process_policy(node, searchSettings->nodePolicyTemperature, isPolicyMap, searchSettings);
    // TODO: assign value + backprop
    node->enable_has_large_nn_results();
}
#endif
