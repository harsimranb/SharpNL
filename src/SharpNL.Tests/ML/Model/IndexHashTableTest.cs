//  
//  Copyright 2014 Gustavo J Knuppe (https://github.com/knuppe)
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

namespace SharpNL.Tests.ML.Model {
    [TestFixture]
    internal class IndexHashTableTest {

        [Test]
        public void testWithCollision() {
            var array = new[] {"7", "21", "0"};

            var arrayIndex = new IndexHashTable<string>(array);

            for (var i = 0; i < array.Length; i++) {
                Assert.AreEqual(i, arrayIndex[array[i]]);
            }

            // has the same slot as as ""
            Assert.AreEqual(-1, arrayIndex["4"]);
        }

        [Test]
        public void testWithoutCollision() {
            var array = new[] {"4", "7", "5"};

            var arrayIndex = new IndexHashTable<string>(array);

            for (var i = 0; i < array.Length; i++) {
                Assert.AreEqual(i, arrayIndex[array[i]]);
            }
        }
    }
}