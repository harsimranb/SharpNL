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
using SharpNL.ML;
using SharpNL.ML.NaiveBayes;
using SharpNL.Utility;

namespace SharpNL.Tests.ML.NaiveBayes {
    [TestFixture]
    public class NaiveBayesPrepAttachTest {

        [Test]
        public void TestNaiveBayesOnPrepAttachData() {

            var model = NaiveBayesTests.TrainModel(PrepAttachDataUtility.CreateTrainingStream());

            Assert.NotNull(model);

            PrepAttachDataUtility.TestModel(model, 0.7897994553107205);
        }

        [Test]
        public void TestNaiveBayesOnPrepAttachDataUsingTrainUtil() {

            var parameters = TrainingParameters.DefaultParameters();
            parameters.Set(Parameters.Algorithm, Parameters.Algorithms.NaiveBayes);
            parameters.Set(Parameters.Cutoff, "1");

            var trainer = TrainerFactory.GetEventTrainer(parameters, null, null);

            Assert.NotNull(trainer);
            Assert.IsInstanceOf<NaiveBayesTrainer>(trainer);

            var model = trainer.Train(PrepAttachDataUtility.CreateTrainingStream());

            Assert.NotNull(model);

            PrepAttachDataUtility.TestModel(model, 0.7897994553107205);
        }

        [Test]
        public void TestNaiveBayesOnPrepAttachDataUsingTrainUtilWithCutoff5() {

            var parameters = TrainingParameters.DefaultParameters();
            parameters.Set(Parameters.Algorithm, Parameters.Algorithms.NaiveBayes);
            parameters.Set(Parameters.Cutoff, "5");

            var trainer = TrainerFactory.GetEventTrainer(parameters, null, null);

            Assert.NotNull(trainer);
            Assert.IsInstanceOf<NaiveBayesTrainer>(trainer);

            var model = trainer.Train(PrepAttachDataUtility.CreateTrainingStream());

            Assert.NotNull(model);

            PrepAttachDataUtility.TestModel(model, 0.7945035899975241);
        }


    }
}