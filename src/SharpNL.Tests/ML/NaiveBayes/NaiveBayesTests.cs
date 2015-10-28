using System.Collections.Generic;
using System.Globalization;
using NUnit.Framework;
using SharpNL.ML.Model;
using SharpNL.ML.NaiveBayes;
using SharpNL.Utility;

namespace SharpNL.Tests.ML.NaiveBayes {
    internal static class NaiveBayesTests {

        internal static NaiveBayesModel TrainModel() {
            var parameters = TrainingParameters.DefaultParameters();
            parameters.Set(Parameters.Cutoff, "1");

            var trainer = new NaiveBayesTrainer();
            trainer.Init(parameters, null);


            return trainer.Train(createTrainingStream());
        }

        internal static NaiveBayesModel TrainModel(IObjectStream<Event> samples, int cutoff = 1) {
            var parameters = TrainingParameters.DefaultParameters();
            parameters.Set(Parameters.Cutoff, cutoff.ToString(CultureInfo.InvariantCulture));

            var trainer = new NaiveBayesTrainer();
            trainer.Init(parameters, null);

            return trainer.Train(samples);
        }

        internal static void TestModel(IMaxentModel model, Event ev, double higherProbability) {
            var outcomes = model.Eval(ev.Context);
            var outcome = model.GetBestOutcome(outcomes);
            Assert.AreEqual(2, outcomes.Length);
            Assert.AreEqual(ev.Outcome, outcome);

            if (ev.Outcome.Equals(model.GetOutcome(0)))
                Assert.AreEqual(higherProbability, outcomes[0], 0.0001);

            if (!ev.Outcome.Equals(model.GetOutcome(0)))
                Assert.AreEqual(1.0 - higherProbability, outcomes[0], 0.0001);

            if (ev.Outcome.Equals(model.GetOutcome(1)))
                Assert.AreEqual(higherProbability, outcomes[1], 0.0001);

            if (!ev.Outcome.Equals(model.GetOutcome(1)))
                Assert.AreEqual(1.0 - higherProbability, outcomes[1], 0.0001);
        }

        internal static IObjectStream<Event> createTrainingStream() {
            var trainingEvents = new List<Event>();

            var label1 = "politics";
            var context1 = new[] { "bow=the", "bow=united", "bow=nations" };
            trainingEvents.Add(new Event(label1, context1));

            var label2 = "politics";
            var context2 = new[] { "bow=the", "bow=united", "bow=states", "bow=and" };
            trainingEvents.Add(new Event(label2, context2));

            var label3 = "sports";
            var context3 = new[] { "bow=manchester", "bow=united" };
            trainingEvents.Add(new Event(label3, context3));

            var label4 = "sports";
            var context4 = new[] { "bow=manchester", "bow=and", "bow=barca" };
            trainingEvents.Add(new Event(label4, context4));

            //return new GenericObjectStream<Event>(trainingEvents);
            return new CollectionObjectStream<Event>(trainingEvents);
        }
    }
}