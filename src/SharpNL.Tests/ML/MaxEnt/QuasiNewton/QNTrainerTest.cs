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
using System.Text;
using NUnit.Framework;
using SharpNL.ML.MaxEntropy.QuasiNewton;
using SharpNL.ML.Model;
using SharpNL.Utility;

namespace SharpNL.Tests.ML.MaxEnt.QuasiNewton {
    [TestFixture]
    internal class QNTrainerTest {
        private const int Iterations = 50;

        [Test]
        public void TestInTinyDevSet() {
            using (
                var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                var trainer = new QNTrainer();

                var model = trainer.TrainModel(Iterations, di);

                Assert.NotNull(model);

                var features2Classify = new[] {
                    "feature2", "feature3", "feature3",
                    "feature3", "feature3", "feature3",
                    "feature3", "feature3", "feature3",
                    "feature3", "feature3", "feature3"
                };
                var eval = model.Eval(features2Classify);

                Assert.NotNull(eval);
            }
        }

        [Test]
        public void TestTrainModelReturnsAQNModel() {
            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                var trainer = new QNTrainer();

                var model = trainer.TrainModel(Iterations, di);

                Assert.NotNull(model);
            }
        }

        [Test]
        public void TestModelEquality() {
            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                var trainer = new QNTrainer();

                var model = trainer.TrainModel(Iterations, di);

                Assert.NotNull(model);
                Assert.True(model.Equals(model));
                Assert.False(model.Equals(null));
            }
        }

        [Test]
        public void TestSerializationModel() {
            using (var eventStream = new RealValueFileEventStream(Tests.GetFullPath(@"/opennlp/data/maxent/real-valued-weights-training-data.txt"), Encoding.UTF8)) {

                var di = new OnePassRealValueDataIndexer(eventStream, 1);

                var trainer = new QNTrainer();

                var model = trainer.TrainModel(Iterations, di);

                Assert.NotNull(model);

                QNModel deserialized;

                using (var mem = new MemoryStream()) {

                    using (var modelWriter = new GenericModelWriter(model, new UnclosableStream(mem))) {
                        modelWriter.Persist();
                        modelWriter.Close();    
                    }

                    mem.Flush();
                    mem.Seek(0, SeekOrigin.Begin);

                    
                    using (var modelReader = new GenericModelReader(new BinaryFileDataReader(mem)))
                        deserialized = modelReader.GetModel() as QNModel;
                    
                }
                Assert.NotNull(deserialized);

                Assert.True(model.Equals(deserialized));

                var features2Classify = new [] {
	                "feature2","feature3", "feature3",
	                "feature3","feature3", "feature3",
	                "feature3","feature3", "feature3",
	                "feature3","feature3", "feature3"
                };

                var eval01 = model.Eval(features2Classify);
                var eval02 = deserialized.Eval(features2Classify);

                Assert.AreEqual(eval01.Length, eval02.Length);
                for (var i = 0; i < eval01.Length; i++)
                    Assert.AreEqual(eval01[i], eval02[i], 0.00000001);

            }
        }
    }
}