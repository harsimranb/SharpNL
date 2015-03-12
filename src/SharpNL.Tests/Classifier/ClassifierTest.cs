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

using System.Collections.Generic;

using SharpNL.Classifier;

using NUnit.Framework;

namespace SharpNL.Tests.Classifier {
    [TestFixture]
    internal class ClassifierTest {

        private class IgnoredResult : AbstractResult<StringClass, string> {
             
        }
        private class IgnoredClassifier : AbstractClassifier<StringClass, IgnoredResult, string> {

            protected override SortedSet<IgnoredResult> Evaluate(string[] features) {

                Assert.AreEqual(3, features.Length);

                Assert.AreEqual("a", features[0]);
                Assert.AreEqual("b", features[1]);
                Assert.AreEqual("c", features[2]);

                return new SortedSet<IgnoredResult>();
            }
        }

        [Test]
        public void IgnoredFeatureTest() {

            var c = new IgnoredClassifier();

            c.Classes.Add(new StringClass("empty") { "nop" });

            c.IgnoredFeatures.Add("1");
            c.IgnoredFeatures.Add("z");

            var r = c.Classify(new[] {"1", "a", "z", "b", "1", "c"});

            Assert.Pass();
        }

        [Test]
        public void WithoutClassesTest() {


            var classifier = new IgnoredClassifier();

            var result = classifier.Classify("without", "classes");

            Assert.NotNull(result);
            Assert.AreEqual(0, result.Count);
        }

    }
}