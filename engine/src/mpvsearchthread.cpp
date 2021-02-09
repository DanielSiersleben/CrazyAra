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

void MPVSearchThread::create_mpv_mini_batch()
{
    Trajectory* tmp_trajectories = nodeQueue->getTrajectories();

    newNodes->setFullData(nodeQueue->getQueue());
    newNodeSideToMove->setFullData(nodeQueue->getSideToMove());
    newTrajectories.insert(newTrajectories.begin(), &tmp_trajectories[0], &tmp_trajectories[searchSettings->largeNetBatchSize]);
}

void MPVSearchThread::set_nn_results_to_child_nodes()
{
    size_t batchIdx = 0;
    for (auto node: *newNodes) {
        if (!node->is_terminal()) {
            fill_mpvnn_results(batchIdx, net->is_policy_map(), valueOutputs, probOutputs, node, tbHits, newNodeSideToMove->get_element(batchIdx), searchSettings);
            node->sort_not_expanded_moves_by_probabilities();
        }
        ++batchIdx;
    }
}

void MPVSearchThread::thread_iteration()
{
    if(nodeQueue->batch_is_ready()){
        create_mpv_mini_batch();

        net->predict(nodeQueue->getInputBuffer(), valueOutputs, probOutputs);

        // now all buffers can be reused
        nodeQueue->mark_batch_completed();

        set_nn_results_to_child_nodes();

        backup_value_outputs();
        newNodeSideToMove->reset_idx();
    }
}

void MPVSearchThread::backup_value_outputs()
{
    for (size_t idx = 0; idx < newNodes->size(); ++idx) {
        Node* node = newNodes->get_element(idx);
#ifdef MCTS_TB_SUPPORT
        const bool solveForTerminal = searchSettings->mctsSolver && node->is_tablebase();
        backup_value<false,true>(node->get_value(), searchSettings->virtualLoss, trajectories[idx], solveForTerminal);
#else
        backup_value<false, true>(node->get_large_net_value(), searchSettings->virtualLoss, newTrajectories[idx], false);
#endif
    }

    newNodes->reset_idx();
    newTrajectories.clear();
}

void MPVSearchThread::set_is_running(bool value)
{
    isRunning = value;
    if(!value){
        newNodes->reset_idx();
        newTrajectories.clear();
        newNodeSideToMove->reset_idx();

        nodeQueue->mpvThread_active->store(false);
        nodeQueue->mark_nodes_as_dequeued();
    }
    else nodeQueue->mpvThread_active->store(true);
}

void fill_mpvnn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, Node *node, size_t& tbHits, SideToMove sideToMove, const SearchSettings* searchSettings)
{
    node->set_probabilities_for_moves(get_policy_data_batch(batchIdx, probOutputs, isPolicyMap), sideToMove);
    node_post_process_policy(node, searchSettings->nodePolicyTemperature, isPolicyMap, searchSettings);
    node->set_large_net_value(valueOutputs[batchIdx]);

    node->enable_has_large_nn_results();
}

#endif
