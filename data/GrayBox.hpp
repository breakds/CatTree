#pragma once

#include <string>
#include <unordered_map>
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
    
    
    inline void zero()
    {
      for ( auto& ele : q ) ele = 0.0;
    }

    void initVoters( std::vector<int> labeled, int level = -1 )
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

    void initVoters( const std::vector<int> &inverseMap,
                     const Bipartite &n_to_l, 
                     const int size )
    {
      int L = n_to_l.sizeB();
      for ( auto& ele : q ) ele = LabelSet::inv;
      for ( int l=0; l<L; l++ ) {
        int count = 0;
        auto _to_n = n_to_l.getFromSet( l );
        for ( auto& ele : _to_n ) {
          if ( ele.first < size ) {
            int n = inverseMap[ele.first];
            count++;
            if ( 1 == count ) {
              for ( int j=0; j<LabelSet::classes; j++ ) {
                q[l*LabelSet::classes+j] = 0.0;
              }
            }
            q[l*LabelSet::classes+dataset.label[n]] += 1.0;
          }
        }
        if ( 0 < count ) {
          scale( &q[l*LabelSet::classes], LabelSet::classes, 1.0/count);
        }
      }
    }

    void initVoters()
    {
      for ( int i=0; i<forest.nodeNum(); i++ ) {
        if ( forest[i].isLeaf() ) {
          double vote[LabelSet::classes];
          memset( vote, 0, sizeof(double) * LabelSet::classes );
          int count = 0;
          for ( auto& ele : forest[i].store ) {
            vote[dataset.label[ele]] += 1.0;
            count++;
          }
          if ( 0 < count ) {
            for ( int k=0; k<LabelSet::classes; k++ ) {
              q[i*LabelSet::classes+k] = vote[k] / count;
            }
          }
        }
      }
    }

    
    void levelDown( std::vector<int>& active ) 
    {
      std::vector<int> tmp;
      for ( auto& ele : active ) {
        for ( auto& c : forest[ele].node->getChild() ) {
          memcpy( &q[c->nodeID*LabelSet::classes], &q[ele*LabelSet::classes], sizeof(double) * LabelSet::classes );
          tmp.push_back( c->nodeID );
        }
      }
      active.swap( tmp );
    }

    int test( const std::vector<int>& idx, 
              const Bipartite &n_to_l, 
              const int begin, 
              const int end,
              bool debug = false ) const
    {
      int count = 0;
      double vote[LabelSet::classes];
      for ( int i=begin; i<end; i++ ) {
        int n = idx[i-begin];
        auto _to_l = n_to_l.getToSet( i );
        memset( vote, 0, sizeof(double) * LabelSet::classes );

        for ( auto& ele : _to_l ) {
          int l = ele.first;
          addto( vote, &q[l * LabelSet::classes], LabelSet::classes );
        }


        int infer = 0;
        for ( int k = 1; k < LabelSet::classes; k++ ) {
          if ( vote[k] > vote[infer] ) infer = k;
        }

        

        if ( debug ) {
          DebugInfo( "True: %d, Infer: %d\n", dataset.label[n], infer );
          printVec( vote, LabelSet::classes );
          ResumeOnRet();
        }
        
        if ( infer == dataset.label[n] ) {
          count++;
        }
      }
      return count;
    }


    int test( std::vector<int>& idx, int level = -1  )
    {
      
      
      
      double vote[LabelSet::classes];
      int count = 0;
      for ( auto& ele : idx ) {
        auto res = forest.query( dataset.feat[ele], level );

        // debugging:
        // if ( hash.end() == hash.find( res[0] ) ) {
        //   hash[res[0]] = std::vector<int>( LabelSet::classes, 0 );
        // }
        // hash[res[0]][dataset.label[ele]]++;

        // debugging:

        memset( vote, 0, sizeof(double) * LabelSet::classes );
        for ( auto& item : res ) {
          addto( vote, &q[item * LabelSet::classes], LabelSet::classes );
        }

        // for ( int i=0; i<LabelSet::classes; i++ ) {
        //   vote[i] *= LabelSet::GetWeight(i);
        // }
        
        int infer = 0;
        for ( int i=1; i<LabelSet::classes; i++ ) {
          if ( vote[i] > vote[infer] ) infer = i;
        }
        if ( infer == dataset.label[ele] ) {
          count++;
        }
      }

      // debugging:
      // for ( auto& ele : hash ) {
      //   const double *vote = &q[ele.first*LabelSet::classes];
      //   int infer = 0;
      //   for ( int i=1; i<LabelSet::classes; i++ ) {
      //     if ( vote[i] * LabelSet::GetWeight(i) > vote[infer] * LabelSet::GetWeight(infer) ) infer = i;
      //   }
      //   DebugInfo( "%d: Vote for %d (%d)", ele.first, infer, ele.second[infer] );
      //   printVec( &ele.second[0], LabelSet::classes );
      // }
      // DebugInfo( "count: %d\n", count );
      // ResumeOnRet();
      
      return count;
    }
  };
}
