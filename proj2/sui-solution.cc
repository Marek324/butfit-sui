#include "memusage.h"
#include "search-strategies.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <set>
#include <stack>

constexpr size_t _50MB = 50 * 1024 * 1024;

struct Node {
  SearchState state;
  SearchAction action;
  std::shared_ptr<Node> parent;

  Node(const SearchState &s, const SearchAction &a,
       std::shared_ptr<Node> p = nullptr)
      : state(s), action(a), parent(p) {};
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

  frontier.push(std::make_shared<Node>(
      init_state, SearchAction(Location(), Location()), nullptr));
  explored.insert(init_state);

  for (auto node = frontier.front(); !frontier.empty(); frontier.pop()) {
    auto state = node->state;

    for (auto &action : state.actions()) {
      if (getCurrentRSS() >= mem_limit_ - _50MB)
        return {};

      auto next = action.execute(state);

      if (explored.count(next))
        continue;
      explored.insert(next);

      auto next_node = std::make_shared<Node>(next, action, node);

      if (next.isFinal())
        return path(next_node);

      frontier.push(next_node);
    }
  }

  return {};
}

size_t depth(std::shared_ptr<Node> &node) {
  auto n = node;
  size_t d = 0;
  while (n->parent != nullptr) {
    d++;
    n = n->parent;
  }
  return d;
}

std::vector<SearchAction>
DepthFirstSearch::solve(const SearchState &init_state) {
  std::stack<std::shared_ptr<Node>> frontier;
  std::set<SearchState> explored;

  frontier.push(std::make_shared<Node>(
      init_state, SearchAction(Location(), Location()), nullptr));
  explored.insert(init_state);

  for (auto node = frontier.top(); !frontier.empty(); frontier.pop()) {
    auto state = node->state;

    for (auto &action : state.actions()) {
      if (getCurrentRSS() >= mem_limit_ - _50MB)
        return {};

      auto next = action.execute(state);

      if (explored.count(next))
        continue;
      explored.insert(next);

      auto next_node = std::make_shared<Node>(next, action, node);

      if (next.isFinal())
        return path(next_node);

      if (depth(next_node) < depth_limit_)
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
  std::set<SearchState> explored;

  frontier.push(
      {compute_heuristic(init_state, *heuristic_),
       std::make_shared<Node>(init_state, SearchAction(Location(), Location()),
                              nullptr)});
  explored.insert(init_state);

  for (auto [_, node] = frontier.top(); !frontier.empty(); frontier.pop()) {
    auto state = node->state;

    for (auto &action : state.actions()) {
      if (getCurrentRSS() >= mem_limit_ - _50MB)
        return {};

      auto next = action.execute(state);

      if (explored.count(next))
        continue;
      explored.insert(next);

      auto next_node = std::make_shared<Node>(next, action, node);

      if (next.isFinal())
        return path(next_node);

      double g = 0;
      double h = compute_heuristic(state, *heuristic_);
      double f = g + h;

      frontier.push({f, next_node});
    }
  }

  return {};
}
