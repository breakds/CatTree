#pragma once

#include <string>
#include "Label.hpp"
#include "RanForest/RanForest.hpp"

using namespace ran_forest;

namespace cat_tree
{
  template<typename dataType = float, template<typename> class splitter = BinaryOnAxis>
  class GrayBox
  {
  public:
    DataSet<dataType> dataset;
    
    Forest<dataType,splitter> forest;

    std::vector<double> q; // voters

    GrayBox() : forest(), q(nullptr) {}

    GrayBox( std::string dataInput, std::string forestDir ) : dataset(dataInput), forest(forestDir)
    {
      // uniform initialization of voters
      q.resize( LabelSet::classes * forest.nodeNum() );
      for ( auto& ele : q ) ele = LabelSet::inv;
    }

    void initVoters( std::vector<int> labeled )
    {
      std::unordered_set<int> set;
      for ( auto& ele : labeled ) set.insert( ele );
      for ( int i=0; i<forest.nodeNum(); i++ ) {
        if ( forest[i].isLeaf() ) {
          double vote[LabelSet::classes];
          memset( vote, 0, sizeof(double) * LabelSet::classes );
          int count = 0;
          for ( auto& ele : forest[i].store ) {
            if ( set.end() != set.find( ele ) ) {
              vote[dataset.label[ele]] += 1.0;
              count++;
            }
          }
          if ( 0 < count ) {
            for ( int k=0; k<LabelSet::classes; k++ ) {
              q[i*LabelSet::classes+k] = vote[k] / count;
            }
          }
        }
      }
    }

    int test( std::vector<int>& idx  )
    {
      double vote[LabelSet::classes];
      int count = 0;
      for ( auto& ele : idx ) {
        auto res = forest.query( dataset.feat[ele] );
        memset( vote, 0, sizeof(double) * LabelSet::classes );
        for ( auto& item : res ) {
          addto( vote, &q[item * LabelSet::classes], LabelSet::classes );
        }
        for ( int i=0; i<LabelSet::classes; i++ ) {
          vote[i] *= LabelSet::GetWeight(i);
        }

        int infer = 0;
        for ( int i=1; i<LabelSet::classes; i++ ) {
          if ( vote[i] > vote[infer] ) infer = i;
        }
        if ( infer == dataset.label[ele] ) {
          count++;
        }
      }
      return count;
    }
  };
}
