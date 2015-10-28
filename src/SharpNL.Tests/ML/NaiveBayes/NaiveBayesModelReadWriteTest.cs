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

using System.IO;
using NUnit.Framework;
using SharpNL.ML.Model;
using SharpNL.ML.NaiveBayes;
using SharpNL.Utility;

namespace SharpNL.Tests.ML.NaiveBayes {
    [TestFixture]
    public class NaiveBayesModelReadWriteTest {

        [Test]
        public void TestBinaryModelPersistence() {
            var model = NaiveBayesTests.TrainModel();

            AbstractModel deserialized;
            using (var data = new MemoryStream()) {

                using (var writer = new BinaryNaiveBayesModelWriter(model, new UnclosableStream(data)))
                    writer.Persist();

                data.Seek(0, SeekOrigin.Begin);

                using (var reader = new BinaryNaiveBayesModelReader(data)) {
                    deserialized = reader.GetModel();
                }

            }

            Assert.NotNull(deserialized);
            Assert.IsInstanceOf<NaiveBayesModel>(deserialized);
        }

        [Test]
        public void TestTextModelPersistence() {
            var model = NaiveBayesTests.TrainModel();

            AbstractModel deserialized;
            using (var data = new MemoryStream()) {

                using (var writer = new PlainTextNaiveBayesModelWriter(model, new UnclosableStream(data)))
                    writer.Persist();

                data.Seek(0, SeekOrigin.Begin);

                using (var reader = new PlainTextNaiveBayesModelReader(data)) {
                    deserialized = reader.GetModel();
                }

            }

            Assert.NotNull(deserialized);
            Assert.IsInstanceOf<NaiveBayesModel>(deserialized);
        }
    }
}