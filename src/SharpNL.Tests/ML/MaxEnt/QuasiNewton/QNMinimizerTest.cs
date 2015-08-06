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

using NUnit.Framework;
using SharpNL.ML.MaxEntropy.QuasiNewton;

namespace SharpNL.Tests.ML.MaxEnt.QuasiNewton {
    [TestFixture]
    public class QNMinimizerTest {

        private const int SafetyTimeout = 5000;

        // The timeout is because there was an infinite loop 
        // while I was developing... Lets keep for safety :P
        [Test, Timeout(SafetyTimeout)] 
        public void testQuadraticFunction() {
            var minimizer = new QNMinimizer();
            var f = new QuadraticFunction();
            var x = minimizer.Minimize(f);
            var minValue = f.ValueAt(x);

            Assert.AreEqual(x[0], 1.0, 0.000001);
            Assert.AreEqual(x[1], 5.0, 0.000001);
            Assert.AreEqual(minValue, 10.0, 0.000001);
        }

        [Test, Timeout(SafetyTimeout)]
        public void testRosenbrockFunction() {
            var minimizer = new QNMinimizer();
            var f = new RosenbrockFunction();
            var x = minimizer.Minimize(f);
            var minValue = f.ValueAt(x);

            Assert.AreEqual(x[0], 1.0, 1e-5);
            Assert.AreEqual(x[1], 1.0, 1e-5);
            Assert.AreEqual(minValue, 0, 1e-10);
        }
    }
}