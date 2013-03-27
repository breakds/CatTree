#pragma once

#include <vector>
#include <memory>
#include "RanForest/RanForest.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "LLPack/algorithms/algebra.hpp"
#include "../data/vector.hpp"

using ran_forest::Bipartite;

namespace cat_tree {

  template <typename dataType = float>
  class TMeanShell
  {
  public:
    // std::vector<dataType> centers;
    std::vector<std::unique_ptr<dataType> > centers;

    struct Options
    {
      int maxIter;
      int replicate;
      double converge;
      int dim;
      Options() : maxIter(100), replicate(10), converge(1e-5), dim(10) {}
    } options;


    TMeanShell() : centers(), options() {}

    void writeCenters( std::string filename )
    {
      WITH_OPEN( out, filename.c_str(), "w" );
      int len = static_cast<int>( centers.size() );
      fwrite( &len, sizeof(int), 1, len );
      for ( auto& ele : centers ) {
        fwrite( &options.dim, sizeof(int), 1, out );
        fwrite( ele.get(), sizeof(dataType), options.dim, out );
      }
      END_WITH( out );
    }


    void readCenters( std::string filename )
    {
      WITH_OPEN( in, filename.c_str(), "r" );
      int len = 0;
      fread( &len, sizeof(int), 1, len );
      centers.resize( len );
      for ( auto& ele : centers ) {
        fread( &options.dim, sizeof(int), 1, in );
        fread( ele.get(), sizeof(dataType), options.dim, in );
      }
      END_WITH( in );
    }

  private:

    template <typename feature_t, template <typename T = feature_t, typename... restArgs> class container>
    void CenterMeans( std::vector<std::unique_ptr<dataType> > &centers,
                      const container<feature_t>& feat,
                      int dim,
                      Bipartite& n_to_l )
    {
      
      int N = n_to_l.sizeA();
      int L = n_to_l.sizeB();
        
      for ( auto& ele : centers ) {
        algebra::zero( ele.get(), dim );
      }
        
      std::vector<int> count( L, 0 );
        
      for ( int n=0; n<N; n++ ) {
        auto& _to_l = n_to_l.from( n );
        for ( auto& ele : _to_l ) {
          int l = ele.first;
          count[l]++;
          for ( int j=0; j<dim; j++ ) {
            centers[l].get()[j] += feat[n][j];
          }
        }
      }
        
      for ( int l=0; l<L; l++ ) {
        if ( 0 < count[l] ) {
          scale( centers[l].get(), dim, static_cast<float>( 1.0 / count[l] ) );
        }
      }
    }
    


  public:

  

    template <typename feature_t, template <typename T = feature_t, typename... restArgs> class container>
    void Clustering( const container<feature_t> &feat,
                     int dim,
                     Bipartite& n_to_l )
    {
      int N = n_to_l.sizeA();
      int L = n_to_l.sizeB();
      
      centers.resize( L );
      for ( auto& ele : centers ) {
        ele.reset( new dataType[dim] );
      }

      options.dim = dim;
      

      CenterMeans( centers, feat, dim, n_to_l );

      Bipartite bimap( N, L );

      double lastEnergy = 0.0;
              
      for ( int iter=0; iter<options.maxIter; iter++ ) {
        
        bimap.clear();
        
        Info( "TMeans iter %d", iter );
        // pick centers
        for ( int n=0; n<N; n++ ) {
          auto& _to_l = n_to_l.from( n );
          heap<double,int> ranker( options.replicate );
          for ( auto& ele : _to_l ) {
            int l = ele.first;
            double dist = 0.0;
            for ( int j=0; j<dim; j++ ) {
              double tmp = centers[l].get()[j] - feat[n][j];
              dist += tmp * tmp;
            }
            ranker.add( dist, l );
          }

          for ( int j=0; j<ranker.len; j++ ) {
            bimap.add( n, ranker[j], 1.0 / options.replicate );
          }
          
        } // end for n

        CenterMeans( centers, feat, dim, bimap );

        // Calculate Energy
        double energy = 0.0;
        for ( int n=0; n<N; n++ ) {
          auto& _to_l = bimap.from( n );
          for ( auto& ele : _to_l ) {
            int l = ele.first;
            for ( int j=0; j<dim; j++ ) {
              double tmp = centers[l].get()[j] - feat[n][j];
              energy += tmp * tmp;
            }
          }
        }
        
        if ( 0 <iter && fabs(lastEnergy-energy) < options.converge ) {
          break;
        }
        
        lastEnergy = energy;
        
        printf( "Energy: %.5lf\n", energy );
        
      } // end for iter

      // update alphas
      for ( int n=0; n<N; n++ ) {
        auto& _to_l = bimap.getSetFrom( n );
        if ( 0 < _to_l.size() ) {

          double s = 0.0;
          for ( auto& ele : _to_l ) {
            int l = ele.first;

            // calculate l2 distance
            double dist = 0.0;
            for ( int j=0; j<dim; j++ ) {
              double tmp = centers[l].get()[j] - feat[n][j];
              dist += tmp * tmp;
            }
            dist = sqrt( dist );

            ele.second = exp( - dist / 0.005 );

            s += ele.second;
          }
          
          s = 1.0 / s;
          for ( auto& ele : _to_l ) {
            ele.second *= s;
          }
        }
      }
      n_to_l = std::move( bimap );
    }


    template <typename feature_t>
    inline void concentrate( const feature_t &p, std::vector<int> &membership )
    {
      heap<double,int> ranker( options.replicate );
      for ( auto& l : membership ) {
        double dist = 0.0;
        for ( int j=0; j<options.dim; j++ ) {
          double tmp = centers[l].get()[j] - p[j];
          dist += tmp * tmp;
        }
        ranker.add( dist, l );
      }
      membership.resize( ranker.len );
      for ( int i=0; i<ranker.len; i++ ) {
        membership[i] = ranker[i];
      }
    }
  };
}
