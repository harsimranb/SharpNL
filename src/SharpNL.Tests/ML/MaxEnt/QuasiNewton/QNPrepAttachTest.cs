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

using System.Diagnostics;
using NUnit.Framework;
using SharpNL.ML;
using SharpNL.ML.MaxEntropy.QuasiNewton;
using SharpNL.ML.Model;
using SharpNL.Utility;

namespace SharpNL.Tests.ML.MaxEnt.QuasiNewton {
    [TestFixture]
    internal class QNPrepAttachTest {

        [Test]
        public void TestQNOnPrepAttachData() {

            var model = new QNTrainer().TrainModel(100,
                new TwoPassDataIndexer(PrepAttachDataUtility.CreateTrainingStream(), 1));

            TestModel(model, 0.8155484030700668);
        }

        [Test]
        public void TestQNOnPrepAttachDataWithParamsDefault() {
            
            var trainParams = new TrainingParameters();

            trainParams.Set(Parameters.Algorithm, Parameters.Algorithms.MaxEntQn);

            var trainer = TrainerFactory.GetEventTrainer(trainParams, null, null);

            var model = trainer.Train(PrepAttachDataUtility.CreateTrainingStream());

            TestModel(model, 0.8115870264917059);

        }

        [Test]
        public void TestQNOnPrepAttachDataWithElasticNetParams() {
            var trainParams = new TrainingParameters();

            trainParams.Set(Parameters.Algorithm, Parameters.Algorithms.MaxEntQn);
            trainParams.Set(Parameters.DataIndexer, Parameters.DataIndexers.TwoPass);
            trainParams.Set(Parameters.Cutoff, "1");
            trainParams.Set(Parameters.L1Cost, "0.25");
            trainParams.Set(Parameters.L2Cost, "1.0");

            var trainer = TrainerFactory.GetEventTrainer(trainParams, null, null);

            var model = trainer.Train(PrepAttachDataUtility.CreateTrainingStream());

            TestModel(model, 0.8229759841544937);
        }

        [Test]
        public void TestQNOnPrepAttachDataWithL1Params() {
            var trainParams = new TrainingParameters();

            trainParams.Set(Parameters.Algorithm, Parameters.Algorithms.MaxEntQn);
            trainParams.Set(Parameters.DataIndexer, Parameters.DataIndexers.TwoPass);
            trainParams.Set(Parameters.Cutoff, "1");
            trainParams.Set(Parameters.L1Cost, "1.0");
            trainParams.Set(Parameters.L2Cost, "0");

            var trainer = TrainerFactory.GetEventTrainer(trainParams, null, null);

            var model = trainer.Train(PrepAttachDataUtility.CreateTrainingStream());

            TestModel(model, 0.8180242634315424);

        }


        [Test]
        public void TestQNOnPrepAttachDataWithL2Params() {
            var trainParams = new TrainingParameters();

            trainParams.Set(Parameters.Algorithm, Parameters.Algorithms.MaxEntQn);
            trainParams.Set(Parameters.DataIndexer, Parameters.DataIndexers.TwoPass);
            trainParams.Set(Parameters.Cutoff, "1");
            trainParams.Set(Parameters.L1Cost, "0");
            trainParams.Set(Parameters.L2Cost, "1.0");

            var trainer = TrainerFactory.GetEventTrainer(trainParams, null, null);

            var model = trainer.Train(PrepAttachDataUtility.CreateTrainingStream());

            TestModel(model, 0.8227283981183461);

        }

        [Test]
        public void TestQNOnPrepAttachDataInParallel() {
            var trainParams = new TrainingParameters();

            trainParams.Set(Parameters.Algorithm, Parameters.Algorithms.MaxEntQn);
            //trainParams.Set(Parameters.Iterations, "100");
            trainParams.Set(Parameters.Threads, "2");

            var trainer = TrainerFactory.GetEventTrainer(trainParams, null, null);

            var model = trainer.Train(PrepAttachDataUtility.CreateTrainingStream());

            TestModel(model, 0.8115870264917059);
        }

        private static void TestModel(IMaxentModel model, double expectedAccuracy) {
            var devEvents = PrepAttachDataUtility.ReadPpaFile(@"devset");

            var total = 0;
            var correct = 0;
            foreach (var ev in devEvents) {

                var targetLabel = ev.Outcome;
                var ocs = model.Eval(ev.Context);

                var best = 0;
                for (var i = 1; i < ocs.Length; i++)
                    if (ocs[i] > ocs[best])
                        best = i;

                var predictedLabel = model.GetOutcome(best);

                if (targetLabel.Equals(predictedLabel))
                    correct++;

                total++;
            }

            var accuracy = correct/(double) total;

            Debug.WriteLine("Accuracy on PPA devset: (" + correct + "/" + total + ") " + accuracy);

            Assert.AreEqual(expectedAccuracy, accuracy, .00001);
        }

    }
}