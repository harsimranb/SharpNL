// 
//  Copyright 2015 Gustavo J Knuppe (https://github.com/knuppe)
// 
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//  
//       http://www.apache.org/licenses/LICENSE-2.0
//  
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
// 
//   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//   - May you do good and not evil.                                         -
//   - May you find forgiveness for yourself and forgive others.             -
//   - May you share freely, never taking more than you give.                -
//   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  

using System;
using NUnit.Framework;
using SharpNL.ML.MaxEntropy.QuasiNewton;

namespace SharpNL.Tests.ML.MaxEnt.QuasiNewton {
    [TestFixture]
    public class LineSearchTest {
        public static readonly double TOLERANCE = 0.01;

        public void testLineSearchFailsAtMinimum2() {
            var objectiveFunction = new QuadraticFunction2();
            // given
            double[] testX = {0};
            var testValueX = objectiveFunction.ValueAt(testX);
            var testGradX = objectiveFunction.GradientAt(testX);
            double[] testDirection = {1};

            // when
            var lsr = LineSearchResult.GetInitialObject(testValueX, testGradX, testX);
            LineSearch.DoLineSearch(objectiveFunction, testDirection, lsr, 1.0);
            var stepSize = lsr.StepSize;

            // then
            Assert.False(TOLERANCE < stepSize && stepSize <= 1);

            Assert.AreEqual(0.0, stepSize, TOLERANCE);
        }

        /// <summary>
        /// Quadratic function: f(x) = (x-2)^2 + 4
        /// </summary>
        private class QuadraticFunction1 : IFunction {
            public int Dimension {
                get { return 1; }
            }

            public double ValueAt(double[] x) {
                return Math.Pow(x[0] - 2, 2) + 4;
            }

            public double[] GradientAt(double[] x) {
                return new[] {2*(x[0] - 2)};
            }
        }

        /// <summary>
        /// Quadratic function: f(x) = x^2
        /// </summary>
        private class QuadraticFunction2 : IFunction {
            public int Dimension {
                get { return 1; }
            }

            public double ValueAt(double[] x) {
                return Math.Pow(x[0], 2);
            }

            public double[] GradientAt(double[] x) {
                return new[] {2*x[0]};
            }
        }

        [Test]
        public void TestLineSearchDeterminesSaneStepLength1() {
            var objectiveFunction = new QuadraticFunction1();
            // given
            var testX = new double[] {0};
            var testValueX = objectiveFunction.ValueAt(testX);
            var testGradX = objectiveFunction.GradientAt(testX);
            var testDirection = new double[] {1};

            // when
            var lsr = LineSearchResult.GetInitialObject(testValueX, testGradX, testX);
            LineSearch.DoLineSearch(objectiveFunction, testDirection, lsr, 1.0);

            var stepSize = lsr.StepSize;

            // then
            Assert.True(TOLERANCE < stepSize && stepSize <= 1);
        }

        [Test]
        public void TestLineSearchDeterminesSaneStepLength2() {
            var objectiveFunction = new QuadraticFunction2();
            // given
            var testX = new double[] {-2};
            var testValueX = objectiveFunction.ValueAt(testX);
            var testGradX = objectiveFunction.GradientAt(testX);
            var testDirection = new double[] {1};

            // when
            var lsr = LineSearchResult.GetInitialObject(testValueX, testGradX, testX);
            LineSearch.DoLineSearch(objectiveFunction, testDirection, lsr, 1.0);
            var stepSize = lsr.StepSize;

            // then
            Assert.True(TOLERANCE < stepSize && stepSize <= 1);
        }

        [Test]
        public void testLineSearchFailsAtMinimum1() {
            var objectiveFunction = new QuadraticFunction2();
            // given
            double[] testX = {0};
            var testValueX = objectiveFunction.ValueAt(testX);
            var testGradX = objectiveFunction.GradientAt(testX);
            double[] testDirection = {-1};
            // when
            var lsr = LineSearchResult.GetInitialObject(testValueX, testGradX, testX);
            LineSearch.DoLineSearch(objectiveFunction, testDirection, lsr, 1.0);
            var stepSize = lsr.StepSize;
            // then
            Assert.False(TOLERANCE < stepSize && stepSize <= 1);

            Assert.AreEqual(0.0, stepSize, TOLERANCE);
        }

        [Test]
        public void testLineSearchFailsWithWrongDirection1() {
            var objectiveFunction = new QuadraticFunction1();
            // given
            double[] testX = {0};
            var testValueX = objectiveFunction.ValueAt(testX);
            var testGradX = objectiveFunction.GradientAt(testX);
            double[] testDirection = {-1};
            // when
            var lsr = LineSearchResult.GetInitialObject(testValueX, testGradX, testX);
            LineSearch.DoLineSearch(objectiveFunction, testDirection, lsr, 1.0);
            var stepSize = lsr.StepSize;
            // then


            Assert.False(TOLERANCE < stepSize && stepSize <= 1);

            Assert.AreEqual(0.0, stepSize, TOLERANCE);
        }

        [Test]
        public void testLineSearchFailsWithWrongDirection2() {
            var objectiveFunction = new QuadraticFunction2();
            // given
            double[] testX = {-2};
            var testValueX = objectiveFunction.ValueAt(testX);
            var testGradX = objectiveFunction.GradientAt(testX);
            double[] testDirection = {-1};

            // when
            var lsr = LineSearchResult.GetInitialObject(testValueX, testGradX, testX);
            LineSearch.DoLineSearch(objectiveFunction, testDirection, lsr, 1.0);
            var stepSize = lsr.StepSize;

            // then
            Assert.False(TOLERANCE < stepSize && stepSize <= 1);

            Assert.AreEqual(0.0, stepSize, TOLERANCE);
        }

        [Test]
        public void testLineSearchFailsWithWrongDirection3() {
            var objectiveFunction = new QuadraticFunction1();
            // given
            double[] testX = {4};
            var testValueX = objectiveFunction.ValueAt(testX);
            var testGradX = objectiveFunction.GradientAt(testX);
            double[] testDirection = {1};

            // when
            var lsr = LineSearchResult.GetInitialObject(testValueX, testGradX, testX);
            LineSearch.DoLineSearch(objectiveFunction, testDirection, lsr, 1.0);
            var stepSize = lsr.StepSize;

            // then
            Assert.False(TOLERANCE < stepSize && stepSize <= 1);

            Assert.AreEqual(0.0, stepSize, TOLERANCE);
        }

        [Test]
        public void testLineSearchFailsWithWrongDirection4() {
            var objectiveFunction = new QuadraticFunction2();
            // given
            double[] testX = {2};
            var testValueX = objectiveFunction.ValueAt(testX);
            var testGradX = objectiveFunction.GradientAt(testX);
            double[] testDirection = {1};

            // when
            var lsr = LineSearchResult.GetInitialObject(testValueX, testGradX, testX);
            LineSearch.DoLineSearch(objectiveFunction, testDirection, lsr, 1.0);
            var stepSize = lsr.StepSize;

            // then
            Assert.False(TOLERANCE < stepSize && stepSize <= 1);

            Assert.AreEqual(0.0, stepSize, TOLERANCE);
        }
    }
}