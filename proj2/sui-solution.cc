#include "memusage.h"
#include "search-strategies.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <set>
#include <stack>

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
  auto curr = node;
  while (curr->parent != nullptr) {
    path.push_back(curr->action);
    curr = curr->parent;
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

  while (!frontier.empty()) {
    auto curr = frontier.front();
    frontier.pop();
    auto state = curr->state;

    for (auto &action : state.actions()) {
      if (getCurrentRSS() >= mem_limit_)
        return {};

      auto next = action.execute(state);

      if (explored.count(next))
        continue;
      explored.insert(next);

      auto node = std::make_shared<Node>(next, action, curr);

      if (next.isFinal())
        return path(node);

      frontier.push(node);
    }
  }

  return {};
}

std::vector<SearchAction>
DepthFirstSearch::solve(const SearchState &init_state) {
  std::stack<std::shared_ptr<Node>> frontier;
  std::set<SearchState> explored;

  frontier.push(std::make_shared<Node>(
      init_state, SearchAction(Location(), Location()), nullptr));
  explored.insert(init_state);

  // depth_limit_
  // mem_limit_

  return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const {
  return 0;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state) {
  std::vector<SearchAction> solution;
  SearchState state(init_state);

  return {};
}
