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
using SharpNL.Tokenize;

namespace SharpNL.Tests.Tokenize {
    [TestFixture]
    internal class TokenUtilitiesTest {

        private static readonly string[] sample = {"a", "a", "A", "b", "c", "c"};

        [Test]
        public void GetTokenCountTest() {
            
            Assert.AreEqual(2, TokenUtilities.GetTokenCount("a", sample, false));
            Assert.AreEqual(3, TokenUtilities.GetTokenCount("a", sample, true));
            Assert.AreEqual(2, TokenUtilities.GetTokenCount("c", sample, false));

        }

        [Test]
        public void GetTokenFrequencyTest() {
            var freq = TokenUtilities.GetTokenFrequency(sample, false);

            Assert.AreEqual(4, freq.Count);

            Assert.AreEqual(2, freq["a"]);
            Assert.AreEqual(1, freq["A"]);
            Assert.AreEqual(1, freq["b"]);
            Assert.AreEqual(2, freq["c"]);
        }

        [Test]
        public void GetTokenFrequencyCaseInsensitiveTest() {
            var freq = TokenUtilities.GetTokenFrequency(sample, true);

            Assert.AreEqual(3, freq["a"]);
            Assert.AreEqual(3, freq["A"]);
        }

        [Test]
        public void GetUniqueTokensCaseInsensitiveTest() {
            var set = TokenUtilities.GetUniqueTokens(sample, true);

            Assert.AreEqual(3, set.Count);
            Assert.True(set.Contains("a"));
            Assert.True(set.Contains("A"));
            Assert.False(set.Contains("x"));
        }
    }
}
