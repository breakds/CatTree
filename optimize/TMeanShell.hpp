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
    std::vector<std::unique_ptr<dataType> > centers;

    struct Options
    {
      int maxIter;
      int replicate;
      double converge;
      int dim;
      double wtBandwidth;
      Options() : maxIter(20), replicate(10), converge(1e-5), dim(10), wtBandwidth(100.0) {}
    } options;


    TMeanShell() : centers(), options() {}

    void writeCenters( std::string filename )
    {
      WITH_OPEN( out, filename.c_str(), "wb" );
      int len = static_cast<int>( centers.size() );
      fwrite( &len, sizeof(int), 1, out );
      for ( auto& ele : centers ) {
        fwrite( &options.dim, sizeof(int), 1, out );
        fwrite( ele.get(), sizeof(dataType), options.dim, out );
      }
      END_WITH( out );
    }


    void readCenters( std::string filename )
    {
      WITH_OPEN( in, filename.c_str(), "rb" );
      int len = 0;
      fread( &len, sizeof(int), 1, in );
      centers.resize( len );
      for ( auto& ele : centers ) {
        fread( &options.dim, sizeof(int), 1, in );
        ele.reset( new dataType[options.dim] );
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
      
      int L = n_to_l.sizeB();

#     pragma omp parallel for
      for ( int l=0; l<L; l++ ) {
        algebra::zero( centers[l].get(), dim );
        auto& _to_n = n_to_l.to( l );
        if ( 0 < _to_n.size() ) {
          int count = 0;
          for ( auto& ele : _to_n ) {
            count++;
            int n = ele.first;
            for ( int j=0; j<dim; j++ ) {
              centers[l].get()[j] += feat[n][j];
            }
          }
          algebra::scale( centers[l].get(), dim, static_cast<float>( 1.0 / count ) );
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
#       pragma omp parallel for
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
          
#         pragma omp critical
          for ( int j=0; j<ranker.len; j++ ) {
            bimap.add( n, ranker[j], 1.0 / options.replicate );
          }
          
        } // end for n

        CenterMeans( centers, feat, dim, bimap );

        // Calculate Energy
        double energy = 0.0;
#       pragma omp parallel for reduction(+ : energy)
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
#     pragma omp parallel for
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

            ele.second = exp( - dist / options.wtBandwidth );
	    
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
    inline void concentrate( const feature_t &p, std::vector<int> &membership ) const
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

    template <typename feature_t>
    inline void concentrate( const feature_t &p, std::vector<std::pair<int,double> > &membership ) const
    {
      heap<double,int> ranker( options.replicate );
      for ( auto& ele : membership ) {
        int l = ele.first;
        double dist = 0.0;
        for ( int j=0; j<options.dim; j++ ) {
          double tmp = centers[l].get()[j] - p[j];
          dist += tmp * tmp;
        }
        ranker.add( sqrt( dist ), l );
      }
      membership.resize( ranker.len );
      for ( int i=0; i<ranker.len; i++ ) {
        membership[i].first = ranker[i];
        membership[i].second = exp( - ranker(i) / options.wtBandwidth );
      }
    }
  };
}
