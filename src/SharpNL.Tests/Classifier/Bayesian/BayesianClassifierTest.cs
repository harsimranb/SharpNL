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

using System.Linq;
using NUnit.Framework;
using SharpNL.Classifier.Bayesian;

namespace SharpNL.Tests.Classifier.Bayesian {
    [TestFixture]
    internal class BayesianClassifierTest {


        [Test]
        public void ClassifierTest() {

            var bc = new BayesianClassifier<int>();

            bc.SetFeatureProbability(10, .50d);
            bc.SetFeatureProbability(20, .2d);
            bc.SetFeatureProbability(30, .50d);
            bc.SetFeatureProbability(69, .99d);

            var set = bc.Classify(new[] { 10, 20, 30, 69 });

            Assert.NotNull(set);
            Assert.AreEqual(1, set.Count);

            var r = set.ElementAt(0);

            Assert.AreEqual(.96d, r.Probability, .009d);

        }

        [Test]
        public void TeachTest() {

            var classifier = new BayesianClassifier<string>(0);

            classifier.TeachMatch("num", new [] { "1", "2", "3", "1" });
            classifier.TeachMatch("num", new [] { "1", "2", "3" });
            classifier.TeachNonMatch("num", new [] { "a", "b", "c" });

            classifier.TeachMatch("chr", new [] { "a", "b", "c" });
            classifier.TeachNonMatch("chr", new [] { "1", "2", "3" });

            var r = classifier.GetBestResult("1");

            Assert.That(r, Is.Not.Null);

            Assert.That(r.Class.Name, Is.EqualTo("num"));

            r = classifier.GetBestResult("c");
            
            Assert.NotNull(r);
            Assert.AreEqual("chr", r.Class.Name);

        }

    }
}