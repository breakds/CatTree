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
  template<typename featType, template<typename> class kernel = VP>
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
    std::vector<size_t> labeled;
    
    // array of testing IDs
    std::vector<size_t> unlabeled;
    
    Forest<dataType,kernel> forest;

    // voters for leaf nodes
    std::vector<std::vector<double> > q;

  public:

    /* ---------- initializations ---------- */
    // default constructor
    BetaBox() : feat(), trueLabel(), labeledp(), labeled(), unlabeled(), forest(), q() {}
    
    void LoadForest( std::string forestDir ) 
    {
      forest.read(forestDir);
      q.resize( forest.numNodes() );
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

    inline void resetVoters()
    {
      for ( auto& ele : q ) {
        std::fill( ele.begin(), ele.end(), LabelSet::inv );
      }
    }


    /* ---------- training ---------- */
    void solve( const Bipartite& graph, int maxIter = 200 )
    {
      size_t numL = labeled.size();
      size_t numU = unlabeled.size();

      std::vector<double> P( numL * LabelSet::classes, 0.0 );
      for ( size_t m=0; m<numL; m++ ) {
        P[ m * LabelSet::classes + trueLabel[m] ] = 1.0;
      }

      std::vector<double> tmpQ( forest.numNodes() * LabelSet::classes );

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

    void directSolve( const Bipartite& graph ) 
    {
      size_t L = graph.sizeB();
      clearVoters();
      std::vector<size_t> count( L, 0 );
      for ( auto& n : labeled ) {
	auto& _to_l = graph.from( n );
	for ( auto& ele : _to_l ) {
	  size_t l = ele.first;
	  q[l][trueLabel[n]] += 1.0;
	  count[l] ++;
	}
      }
      
      for ( size_t l=0; l<L; l++ ) {
	if ( 0 < count[l] ) {
	  algebra::scale( &q[l][0], LabelSet::classes, 1.0 / count[l] );
	}
      }
    }
    

    /* ---------- properties/accessors ---------- */
    inline int dim() const
    {
      return feat[0].dim();
    }
    
    size_t size() const
    {
      return feat.size();
    }

    /* ---------- testing ---------- */
  };
}

