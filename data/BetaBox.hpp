#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "PatTk/data/Label.hpp"
#include "LLPack/utils/candy.hpp"
#include "RanForest/RanForest.hpp"
#include "../optimize/power_solver.hpp"


using namespace ran_forest;
using namespace PatTk;

namespace cat_tree
{
  template<typename featType, template<typename> class splitter = BinaryOnDistance>
  class BetaBox
  {
  public:

    typedef typename ElementOf<featType>::type dataType;
    

    // array of feature vectors
    std::vector<featType> feat;

    // array of ground truth labels
    std::vector<int> trueLabel;

    // array of traning/testing indicators
    std::vector<bool> labeledp;

    // array of training IDs
    std::vector<int> labeled;
    
    // array of testing IDs
    std::vector<int> unlabeled;
    
    Forest<dataType,splitter> forest;

    // voters for leaf nodes
    std::vector<std::vector<double> > q;

  public:

    /* ---------- initializations ---------- */
    // default constructor
    BetaBox() : feat(), trueLabel(), labeledp(), labeled(), unlabeled(), forest(), q() {}
    
    void LoadForest( std::string forestDir ) 
    {
      forest.read(forestDir);
      q.resize( forest.nodeNum() );
      for ( auto& ele : q ) {
        ele.resize( LabelSet::classes );
        std::fill( ele.begin(), ele.end(), LabelSet::inv );
      }
      forest.Summary();
    }

    inline void clearVoters()
    {
      for ( auto& ele : q ) {
        std::fill( ele.begin(), ele.end(), 0.0 );
      }
    }


    /* ---------- training ---------- */
    void solve( const Bipartite& graph, int maxIter = 200 )
    {
      int numL = static_cast<int>( labeled.size() );
      int numU = static_cast<int>( unlabeled.size() );

      std::vector<double> P( numL * LabelSet::classes, 0.0 );
      for ( int m=0; m<numL; m++ ) {
        P[ m * LabelSet::classes + trueLabel[m] ] = 1.0;
      }

      std::vector<double> tmpQ( forest.nodeNum() * LabelSet::classes );

      std::unique_ptr<double> y;
      
      PowerSolver solve;
      solve.options.powerMaxIter = maxIter;
      solve( numL, numU, &P[0], &graph, &tmpQ[0], y );

      auto p = tmpQ.begin();
      for ( auto& ele : q ) {
        for ( auto& item : ele ) {
          item = *(p++);
        }
      }
    }
    

    /* ---------- properties/accessors ---------- */
    inline int dim() const
    {
      return feat[0].dim();
    }
    
    int size() const
    {
      return static_cast<int>( feat.size() );
    }

    /* ---------- testing ---------- */
  };
}

