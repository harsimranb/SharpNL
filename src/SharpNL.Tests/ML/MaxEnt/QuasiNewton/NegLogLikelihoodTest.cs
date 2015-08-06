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
using System.Collections;
using System.Text;
using NUnit.Framework;
using SharpNL.ML.MaxEntropy.QuasiNewton;
using SharpNL.ML.Model;

namespace SharpNL.Tests.ML.MaxEnt.QuasiNewton {
    [TestFixture]
    public class NegLogLikelihoodTest {

        private static readonly double Tolerance1 = 1.0E-06;
        private static readonly double Tolerance2 = 1.0E-10;

        [Test]
        public void TestDomainDimensionSanity() {

            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {
                
                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                di.Execute();
                
                var func = new NegLogLikelihood(di);

                var correctDomainDimension = di.GetPredLabels().Length*di.GetOutcomeLabels().Length;

                Assert.AreEqual(correctDomainDimension, func.Dimension);
            }
        }

        [Test]
        public void TestInitialSanity() {
            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                di.Execute();

                var func = new NegLogLikelihood(di);

                var initial = func.GetInitialPoint();

                // ReSharper disable once ForCanBeConvertedToForeach
                for (var i = 0; i < initial.Length; i++) {
                    Assert.AreEqual(0, initial[i], Tolerance1);
                }
            }
        }

        [Test]
        public void TestGradientSanity() {
            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                di.Execute();

                var func = new NegLogLikelihood(di);

                var initial = func.GetInitialPoint();
                var gradientAtInitial = func.GradientAt(initial);

                // then
                Assert.NotNull(gradientAtInitial);
            }

        }

        [Test]
        public void TestValueAtInitialPoint() {
            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                di.Execute();

                var func = new NegLogLikelihood(di);

                const double expectedValue = 13.86294361;

                var value = func.ValueAt(func.GetInitialPoint());

                Assert.AreEqual(expectedValue, value, Tolerance1);
            }
        }

        [Test]
        public void TestValueAtNonInitialPoint01() {
            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                di.Execute();

                var func = new NegLogLikelihood(di);

                double[] nonInitialPoint = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
                var value = func.ValueAt(nonInitialPoint);
                const double expectedValue = 13.862943611198894;

                Assert.AreEqual(expectedValue, value, Tolerance1);
            }
        }

        [Test]
        public void TestValueAtNonInitialPoint02() {
            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                di.Execute();

                var func = new NegLogLikelihood(di);

                var nonInitialPoint = new double[] { 3, 2, 3, 2, 3, 2, 3, 2, 3, 2 };
                var value = func.ValueAt(DealignDoubleArrayForTestData(nonInitialPoint,
                        di.GetPredLabels(),
                        di.GetOutcomeLabels()));
                const double expectedValue = 53.163219721099026;

                Assert.AreEqual(expectedValue, value, Tolerance2);
            }
        }

        [Test]
        public void TestGradientAtInitialPoint() {
            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                di.Execute();

                var func = new NegLogLikelihood(di);

                var nonInitialPoint = new[] { 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5 };
                var gradientAtNonInitialPoint = func.GradientAt(DealignDoubleArrayForTestData(nonInitialPoint, di.GetPredLabels(), di.GetOutcomeLabels()));
                var expectedGradient = 
                        new[] { -12.755042847945553, -21.227127506102434,
                               -72.57790706276435,   38.03525795198456,
                                15.348650889354925,  12.755042847945557,
                                21.22712750610244,   72.57790706276438,
                               -38.03525795198456,  -15.348650889354925 };

                Assert.True(CompareDoubleArray(expectedGradient, gradientAtNonInitialPoint, di, Tolerance1));
            }
        }

        [Test]
        public void TestGradientAtNonInitialPoint() {
            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                di.Execute();

                var func = new NegLogLikelihood(di);

                var gradientAtInitialPoint = func.GradientAt(func.GetInitialPoint());
                var expectedGradient = new[] { -9.0, -14.0, -17.0, 20.0, 8.5, 9.0, 14.0, 17.0, -20.0, -8.5 };

                Assert.True(CompareDoubleArray(expectedGradient, gradientAtInitialPoint, di, Tolerance1));
            }
        }

        private static double[] AlignDoubleArrayForTestData(double[] expected, string[] predLabels, string[] outcomeLabels) {
            double[] aligned = new double[predLabels.Length * outcomeLabels.Length];

            var sortedPredLabels = (string[])predLabels.Clone();
            var sortedOutcomeLabels = (string[])outcomeLabels.Clone();

            Array.Sort(sortedPredLabels);
            Array.Sort(sortedOutcomeLabels);

            var invertedPredIndex = new Hashtable();
            var invertedOutcomeIndex = new Hashtable();
            for (var i = 0; i < predLabels.Length; i++) {
                invertedPredIndex.Add(predLabels[i], i);
            }
            for (var i = 0; i < outcomeLabels.Length; i++) {
                invertedOutcomeIndex.Add(outcomeLabels[i], i);
            }

            for (var i = 0; i < sortedOutcomeLabels.Length; i++) {
                for (var j = 0; j < sortedPredLabels.Length; j++) {
                    aligned[i * sortedPredLabels.Length + j] = expected[(int)invertedOutcomeIndex[sortedOutcomeLabels[i]]
                        * sortedPredLabels.Length
                        + (int)invertedPredIndex[sortedPredLabels[j]]];
                }
            }

            return aligned;
        }

        private static double[] DealignDoubleArrayForTestData(double[] expected, string[] predLabels, string[] outcomeLabels) {
            var dealigned = new double[predLabels.Length * outcomeLabels.Length];

            var sortedPredLabels = (string[])predLabels.Clone();
            var sortedOutcomeLabels = (string[])outcomeLabels.Clone();

            Array.Sort(sortedPredLabels);
            Array.Sort(sortedOutcomeLabels);

            var invertedPredIndex = new Hashtable();
            var invertedOutcomeIndex = new Hashtable();
            for (var i = 0; i < predLabels.Length; i++) {
                invertedPredIndex.Add(predLabels[i], i);
            }
            for (var i = 0; i < outcomeLabels.Length; i++) {
                invertedOutcomeIndex.Add(outcomeLabels[i], i);
            }

            for (var i = 0; i < sortedOutcomeLabels.Length; i++) {
                for (var j = 0; j < sortedPredLabels.Length; j++) {
                    dealigned[(int)invertedOutcomeIndex[sortedOutcomeLabels[i]]
                        * sortedPredLabels.Length
                        + (int)invertedPredIndex[sortedPredLabels[j]]] = expected[i
                        * sortedPredLabels.Length + j];
                }
            }

            return dealigned;
        }

        private bool CompareDoubleArray(double[] expected, double[] actual, IDataIndexer indexer, double tolerance) {
            var alignedActual = AlignDoubleArrayForTestData(actual, indexer.GetPredLabels(), indexer.GetOutcomeLabels());

            if (expected.Length != alignedActual.Length)
                return false;

            for (var i = 0; i < alignedActual.Length; i++)
                if (Math.Abs(alignedActual[i] - expected[i]) > tolerance)
                    return false;
                
            return true;
        }

    }
}