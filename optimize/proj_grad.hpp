#pragma once

#include <functional>



namespace cat_tree {
  template <typename storeType>
  class ProjGradSolver
  {
  private:
    // assistant vectors
    std::vector<double> t1, t2;
    
  public:

    struct Options
    {
      double converge;
      int maxIter;
      // for line search
      double lsTh;
      double lsTurns;
      // for display
      bool iterEnergyInfo;
      
      Options() : converge( 1e-5 ), maxIter( 20 ), lsTh(1e-8), lsTurns(40),
                  iterEnergyInfo( true )
      {}
    } options;
    
    std::vector<double> x;
    
    storeType store;

    ProjGradSolver() {}
    
    explicit ProjGradSolver( size_t d, std::function<void(std::vector<double>&,storeType&)> init )
    {
      x.resize( d, 0.0 );
      init( x, store );
    }


    double operator()( std::function<double(const std::vector<double>&,storeType&)> energy,
                     std::function<void(std::vector<double>&, const std::vector<double>&,storeType&)> negderiv,
                     std::function<void(std::vector<double>&, storeType&)> projection )
    {
      // initialize assitant vectors for line search
      t1.resize( x.size() );
      t2.resize( x.size() );
      double last_e = energy( x, store );
      std::vector<double> d( x.size() );
      
      if ( options.iterEnergyInfo ) {
        Info( "iter 0: %.6lf", last_e );
      }
      
      for ( int iter=0; iter<options.maxIter; iter++ ) {
        
        // negative derivative
        negderiv( d, x, store );

        
        // line search
        double a = 1.0;
        bool updated = false;
        double dE2 = algebra::norm2( d, x.size() );
        if ( dE2 < options.lsTh ) {
          return last_e;
        }
        double E0 = energy( x, store );
        
        // t1 = x + a * d
        algebra::copy( t1, x, x.size() );
        algebra::addScaledTo( t1, d, x.size(), a );
        
        // t2 = proj(t1)
        projection( t1, store );
        
        double E_a = energy( t1, store );
        
        if ( E_a >= E0 ) {
          // Shrinking branch
          for ( int i=0; i<options.lsTurns; i++ ) {
            a = ( a * a ) * dE2 / ( 2 * ( E_a - (E0 - dE2 * a ) ) );
            
            // update E_a 
            algebra::copy( t1, x, x.size() );
            algebra::addScaledTo( t1, d, x.size(), a );
            projection( t1, store );
            E_a = energy( t1, store );
            
            if ( E_a < E0 ) {
              updated = true;
              break;
            }
          }
          
        } else { 
          // Expanding branch

          updated = true;

          double E_best = E_a;

          for ( int i=0; i<options.lsTurns; i++ ) {
            if ( E_a > E0 - dE2 * a * ( 1 - 0.5 / 2.0 ) ) {
              double b = ( a * a ) * dE2 / ( 2 * ( E_a - (E0 - dE2 * a ) ) );

              algebra::copy( t2, x, x.size() );
              algebra::addScaledTo( t2, d, x.size(), b );
              projection( t2, store );
              E_a = energy( t2, store );

              if ( E_a < E_best ) {
                E_best = E_a;
                algebra::copy( t1, t2, x.size() );
              } else {
                a *= 2.0;
              }
              
              algebra::copy( t2, x, x.size() );
              algebra::addScaledTo( t2, d, x.size(), a );
              projection( t2, store );
              E_a = energy( t2, store );
              
              
              if ( E_a < E_best ) {
                E_best = E_a;
                algebra::copy( t1, t2, x.size() );
              } else {
                break;
              }
            }
          }
        } // end else
                          
        
        if ( updated ) {
          algebra::copy( x, t1, x.size() );
          double e = energy( x, store );
          if ( options.iterEnergyInfo ) {
            Info( "iter %d: %.6lf", iter + 1, e );
          }
          if ( fabs( e - last_e ) < options.converge ) {
            return e;
          } else {
            last_e = e;
          }
        } else {
          break;
        }

        
      }

      return last_e;
    }
  };
}
