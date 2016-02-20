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
using SharpNL.SentenceDetector;

namespace SharpNL.Tests.Sentence {
    [TestFixture]
    internal class NewlineSentenceDetectorTest {

        private static void TestSentences(string sentences) {
            var sd = new NewlineSentenceDetector();

            var result = sd.SentDetect(sentences);

            Assert.AreEqual(3, result.Length);
            Assert.AreEqual("one.", result[0]);
            Assert.AreEqual("two.", result[1]);
            Assert.AreEqual("three.", result[2]);
        }

        [Test]
        public void NewlineCrTest() {
            TestSentences("one.\rtwo. \r\r three.\r");
        }

        [Test]
        public void NewlineLfTest() {
            TestSentences("one.\ntwo. \n\n three.\n");
        }

        [Test]
        public void NewlineCrLfTest() {
            TestSentences("one.\r\ntwo. \r\n\r\n three.\r\n");
        }

    }
}