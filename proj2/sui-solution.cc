#include "search-strategies.h"

std::vector<SearchAction>
BreadthFirstSearch::solve(const SearchState &init_state) {
  std::vector<SearchAction> solution;
  SearchState curr(init_state);

  return solution;
}

std::vector<SearchAction>
DepthFirstSearch::solve(const SearchState &init_state) {
  std::vector<SearchAction> solution;
  SearchState curr(init_state);

  return solution;
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const {
  return 0;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state) {
  std::vector<SearchAction> solution;
  SearchState curr(init_state);

  return solution;
}
