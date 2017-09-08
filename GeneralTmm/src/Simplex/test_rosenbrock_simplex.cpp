/**
 * \file test_rosenbrock_simplex.cpp
 */

#include <iostream>
#include <Eigen/Core>

#include "simplex.h"
#include "criteria.h"

typedef Eigen::Vector2f ParameterTypeRosenbrock;

// Rosenbrock cost function
struct Rosenbrock
{
  typedef float DataType;
  typedef ParameterTypeRosenbrock ParameterType;
  float operator()(const ParameterTypeRosenbrock& parameters) const
  {
    return (parameters(1, 0) - parameters(0, 0) * parameters(0, 0)) * (parameters(1, 0) - parameters(0, 0) * parameters(0, 0))
        + (1 - parameters(0, 0)) * (1 - parameters(0, 0));
  }
};

int main(int argc, char** argv)
{
  ParameterTypeRosenbrock start_point;
  start_point << 10, 10;

  Rosenbrock fun;

  long max_iterations = 1000;
  float ftol = 0.00001;

  auto optimizer = Optimization::Local::build_simplex( // Builder to generate the correct optimizer
      fun,                                             // Used to infer function type
      Optimization::Local::make_and_criteria(Optimization::Local::IterationCriterion(max_iterations),
          Optimization::Local::RelativeValueCriterion<float>(ftol))); // Compoiste stoping criterion

  optimizer.set_start_point(start_point);  // Starting parameters
  optimizer.set_delta(1);                  // Simplex size
  optimizer.optimize(fun);                 // Optimization start

  std::cout << "Starting point: " << start_point << std::endl;
  std::cout << "Starting value: " << fun(start_point) << std::endl;
  std::cout << "Best point: " << optimizer.get_best_parameters() << std::endl; // Retrieve the best parameters
  std::cout << "Best value: " << optimizer.get_best_value() << std::endl;      // Retrieve the best value
}

