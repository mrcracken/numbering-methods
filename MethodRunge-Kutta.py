"""
@author: mrcraken
@since: 2018
"""

import numpy as np
import platform
  
def rk4 ( t0, u0, dt, f ):

  """
  RK4 takes one Runge-Kutta step.

  Discussion:

    It is assumed that an initial value problem, of the form

      du/dt = f ( t, u )
      u(t0) = u0

    is being solved.

    If the user can supply current values of t, u, a stepsize dt, and a
    function to evaluate the derivative, this function can compute the
    fourth-order Runge Kutta estimate to the solution at time t+dt.

  Parameters:

    Input, real T0, the current time.

    Input, real U0, the solution estimate at the current time.

    Input, real DT, the time step.

    Input, function value = F ( T, U ), a function which evaluates
    the derivative, or right hand side of the problem.

    Output, real U1, the fourth-order Runge-Kutta solution estimate
    at time T0+DT.

  Get four sample values of the derivative.
  """
  f1 = f ( t0,            u0 )
  f2 = f ( t0 + dt / 2.0, u0 + dt * f1 / 2.0 )
  f3 = f ( t0 + dt / 2.0, u0 + dt * f2 / 2.0 )
  f4 = f ( t0 + dt,       u0 + dt * f3 )
#
#  Combine them to estimate the solution U1 at time T1 = T0 + DT.
#
  u1 = u0 + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0

  return u1

def rk4_test ( ):

  print ( '' )
  print ( 'RK4_TEST' )
  print ( '  RK4 takes one Runge-Kutta step for a scalar ODE.' )

  print ( '' )
  print ( '          T          U(T)' )
  print ( '' )

  dt = 0.1
  t0 = 0.0
  tmax = 4 * np.pi
  u0 = 1

  t_num = int ( 2 + ( tmax - t0 ) / dt )

  t = np.zeros ( t_num )
  u = np.zeros ( t_num )

  i = 0
  t[0] = t0
  u[0] = u0

  while ( True ):
#
#  Print (T0,U0).
#
    print ( '  %4d  %14.6f  %14.6g' % ( i, t0, u0 ) )
#
#  Stop if we've exceeded TMAX.
#
    if ( tmax <= t0 ):
      break
#
#  Otherwise, advance to time T1, and have RK4 estimate 
#  the solution U1 there.
#
    t1 = t0 + dt
    u1 = rk4 ( t0, u0, dt, rk4_test_f )

    i = i + 1
    t[i] = t1
    u[i] = u1
    """
    Shift the data to prepare for another step.
    """
    t0 = t1
    u0 = u1
#
#  Terminate.
#
  print ( '' )
  print ( 'Rk4_TEST:' )
  print ( '  Normal end of execution.' )
  return

def rk4_test_f ( t, u ):

    """
    RK4_TEST_F evaluates the right hand side of a particular ODE.
    
    Licensing:
    
    This code is distributed under the GNU LGPL license. 
    
    Modified:
    
    Parameters:
    
    Input, real T, the current time.
    
    Input, real U, the current solution value.
    
    Output, real VALUE, the value of the derivative, dU/dT.
    """ 

    value = -np.sin ( u )
  
    return value

def rk4_tests ( ):

  """
   RK4_TESTS tests the RK4 library.
  """
  print ( '' )
  print ( 'RK4_TESTS:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Test the RK4 library.' )


  rk4_test ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'RK4_TESTS:' )
  print ( '  Normal end of execution.' )
  return

def timestamp ( ):

  """
  TIMESTAMP prints the date as a timestamp.
  """
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return None

def timestamp_test ( ):

  """
  TIMESTAMP_TEST tests TIMESTAMP.
  """

  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  TIMESTAMP prints a timestamp of the current date and time.' )
  print ( '' )

  timestamp ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  timestamp ( )
  rk4_tests ( )
  timestamp ( )
