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
using SharpNL.ML.Model;
using SharpNL.ML.NaiveBayes;

namespace SharpNL.Tests.ML.NaiveBayes {
    [TestFixture]
    public class NaiveBayesCorrectnessTest {

        [TestFixtureSetUp]
        public void Setup() {
            // Naive Bayes should always be run with smoothing, taken out here for mathematical verification
            NaiveBayesModel.Smoothed = false;
        }

        [TestFixtureTearDown]
        public void TearDown() {
            // Turning smoothing back on to avoid interfering with other tests
            NaiveBayesModel.Smoothed = true; 
        }

        [Test]
        public void TestNaiveBayes1() {
            var model = NaiveBayesTests.TrainModel();
            var label = "politics";
            var context = new[] {"bow=united", "bow=nations"};
            var e = new Event(label, context);

            NaiveBayesTests.TestModel(model, e, 1.0);
        }

        [Test]
        public void TestNaiveBayes2() {
            var model = NaiveBayesTests.TrainModel();
            var label = "sports";
            var context = new[] { "bow=manchester", "bow=united" };
            var e = new Event(label, context);

            NaiveBayesTests.TestModel(model, e, 1.0);
        }

        [Test]
        public void TestNaiveBayes3() {
            var model = NaiveBayesTests.TrainModel();
            var label = "politics";
            var context = new[] { "bow=united" };
            var e = new Event(label, context);

            NaiveBayesTests.TestModel(model, e, 2.0 / 3.0);
        }

        [Test]
        public void TestNaiveBayes4() {
            var model = NaiveBayesTests.TrainModel();
            var label = "politics";
            var context = new string[0];
            var e = new Event(label, context);

            NaiveBayesTests.TestModel(model, e, 7.0 / 12.0);
        }

    }
}