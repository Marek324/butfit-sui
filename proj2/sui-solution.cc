#include "memusage.h"
#include "search-strategies.h"

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stack>

constexpr size_t _50MB = 50 * 1024 * 1024;

struct Node {
  SearchState state;
  SearchAction action;
  int depth;
  std::shared_ptr<Node> parent;

  Node(const SearchState &s, const SearchAction &a, int d,
       std::shared_ptr<Node> p = nullptr)
      : state(s), action(a), depth(d), parent(p) {};
};

std::vector<SearchAction> path(std::shared_ptr<Node> &node) {
  std::vector<SearchAction> path;
  auto n = node;
  while (n->parent != nullptr) {
    path.push_back(n->action);
    n = n->parent;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<SearchAction>
BreadthFirstSearch::solve(const SearchState &init_state) {
  std::queue<std::shared_ptr<Node>> frontier;
  std::set<SearchState> explored;

  if (init_state.isFinal())
    return {};

  frontier.push(std::make_shared<Node>(
      init_state, SearchAction(Location(), Location()), 0, nullptr));
  explored.insert(init_state);

  while (!frontier.empty()) {
    auto node = frontier.front();
    frontier.pop();
    auto state = node->state;

    for (auto &action : state.actions()) {
      if (getCurrentRSS() >= mem_limit_ - _50MB)
        return {};

      auto next = action.execute(state);

      if (explored.count(next))
        continue;
      explored.insert(next);

      auto next_node = std::make_shared<Node>(next, action, 0, node);

      if (next.isFinal())
        return path(next_node);

      frontier.push(next_node);
    }
  }

  return {};
}
std::vector<SearchAction>
DepthFirstSearch::solve(const SearchState &init_state) {
  std::stack<std::shared_ptr<Node>> frontier;
  std::set<SearchState> explored;

  if (init_state.isFinal())
    return {};

  frontier.push(std::make_shared<Node>(
      init_state, SearchAction(Location(), Location()), 0, nullptr));
  explored.insert(init_state);

  while (!frontier.empty()) {
    auto node = frontier.top();
    frontier.pop();
    auto state = node->state;

    int next_depth = node->depth + 1;
    for (auto &action : state.actions()) {
      if (getCurrentRSS() >= mem_limit_ - _50MB)
        return {};

      auto next = action.execute(state);

      if (explored.count(next))
        continue;
      explored.insert(next);

      auto next_node = std::make_shared<Node>(next, action, next_depth, node);

      if (next.isFinal())
        return path(next_node);

      if (next_depth < depth_limit_)
        frontier.push(next_node);
    }
  }

  return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const {
  return 0;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state) {
  std::priority_queue<std::pair<double, std::shared_ptr<Node>>,
                      std::vector<std::pair<double, std::shared_ptr<Node>>>,
                      std::greater<>>
      frontier;
  std::map<SearchState, double> explored;

  if (init_state.isFinal())
    return {};

  frontier.push(
      {compute_heuristic(init_state, *heuristic_),
       std::make_shared<Node>(init_state, SearchAction(Location(), Location()),
                              0, nullptr)});
  explored[init_state] = 0;

  while (!frontier.empty()) {
    auto [_, node] = frontier.top();
    frontier.pop();
    auto state = node->state;

    int next_depth = node->depth + 1;
    for (auto &action : state.actions()) {
      if (getCurrentRSS() >= mem_limit_ - _50MB)
        return {};

      auto next = action.execute(state);
      auto next_node = std::make_shared<Node>(next, action, next_depth, node);

      if (next.isFinal())
        return path(next_node);

      if (explored.count(next) && explored[next] <= next_depth)
        continue;
      explored[next] = next_depth;

      double h = compute_heuristic(next, *heuristic_);
      double f = next_depth + h;

      frontier.push({f, next_node});
    }
  }

  return {};
}
