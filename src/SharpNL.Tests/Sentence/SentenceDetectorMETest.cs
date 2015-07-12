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

using System;
using System.IO;
using NUnit.Framework;
using SharpNL.SentenceDetector;
using SharpNL.Utility;

namespace SharpNL.Tests.Sentence {
    [TestFixture]
    internal class SentenceDetectorMETest {

        [Test]
        public void TestSentenceDetector() {
            using (var file = Tests.OpenFile("/opennlp/tools/sentdetect/Sentences.txt")) {

                var mlParams = new TrainingParameters();

                mlParams.Set(Parameters.Iterations, "100");
                mlParams.Set(Parameters.Cutoff, "0");

                var sdFactory = new SentenceDetectorFactory("en", true, null, null);
                var stream = new SentenceSampleStream(new PlainTextByLineStream(file));

                var model = SentenceDetectorME.Train("en", stream, sdFactory, mlParams);

                Assert.AreEqual("en", model.Language);
                Assert.AreEqual(model.UseTokenEnd, true);

                EvalSentences(new SentenceDetectorME(model));
            }
        }

        [Test]
        public void AbbreviationDefaultBehaviorTest() {

            var samples =
                "Test E-mail met zowel letsel als 12. Toedracht in het onderwerp." + Environment.NewLine +
                "Dit is een 2e regel met een tel. 011-4441444 erin." + Environment.NewLine +
                "Dit is een 2e regel." + Environment.NewLine +
                "Dit is een 2e regel." + Environment.NewLine + Environment.NewLine +

                "Dit is een 2e regel met een tel. 033-1333123 erin!" + Environment.NewLine +
                "Test E-mail met zowel winst als 12. Toedracht in het onderwerp." + Environment.NewLine +
                "Dit is een 2e regel!" + Environment.NewLine +
                "Dit is een 2e regel." + Environment.NewLine;

            var stringsToIgnoreDictionary = new SharpNL.Dictionary.Dictionary(true) {
                {"12. Toedracht"},
                {"Tel."},
            };

            var trainingParameters = new TrainingParameters();

            trainingParameters.Set(Parameters.Algorithm, "MAXENT");
            trainingParameters.Set(Parameters.TrainerType, "Event");
            trainingParameters.Set(Parameters.Iterations, "100");
            trainingParameters.Set(Parameters.Cutoff, "5");

            char[] eos = { '.', '?', '!' };
            var sdFactory = new SentenceDetectorFactory("nl", true, stringsToIgnoreDictionary, eos);
            var stringReader = new StringReader(samples);
            var stream = new SentenceSampleStream(new PlainTextByLineStream(stringReader));

            var sentenceModel = SentenceDetectorME.Train("nl", stream, sdFactory, trainingParameters);
            var sentenceDetectorMe = new SentenceDetectorME(sentenceModel);

            var sentences = sentenceDetectorMe.SentDetect(samples);
            var expected = samples.Split(new []{ Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries);


            Assert.AreEqual(8, sentences.Length);
            for (int i = 0; i < sentences.Length; i++) {
                Assert.AreEqual(expected[i], sentences[i]);
            }
        }

        [Test]
        public void TestWithOpenNLPModel() {
            using (var file = Tests.OpenFile("/opennlp/models/en-sent.bin")) {

                var model = new SentenceModel(file);
                EvalSentences(new SentenceDetectorME(model));
            }
        }

        internal static void EvalSentences(SentenceDetectorME sentDetect) {
            const string sampleSentences1 = "This is a test. There are many tests, this is the second.";
            var sents = sentDetect.SentDetect(sampleSentences1);
            Assert.AreEqual(sents.Length, 2);
            Assert.AreEqual(sents[0], "This is a test.");
            Assert.AreEqual(sents[1], "There are many tests, this is the second.");
            var probs = sentDetect.GetSentenceProbabilities();
            Assert.AreEqual(probs.Length, 2);

            const string sampleSentences2 = "This is a test. There are many tests, this is the second";
            sents = sentDetect.SentDetect(sampleSentences2);
            Assert.AreEqual(sents.Length, 2);
            probs = sentDetect.GetSentenceProbabilities();
            Assert.AreEqual(probs.Length, 2);
            Assert.AreEqual(sents[0], "This is a test.");
            Assert.AreEqual(sents[1], "There are many tests, this is the second");

            const string sampleSentences3 = "This is a \"test\". He said \"There are many tests, this is the second.\"";
            sents = sentDetect.SentDetect(sampleSentences3);
            Assert.AreEqual(sents.Length, 2);
            probs = sentDetect.GetSentenceProbabilities();
            Assert.AreEqual(probs.Length, 2);
            Assert.AreEqual(sents[0], "This is a \"test\".");
            Assert.AreEqual(sents[1], "He said \"There are many tests, this is the second.\"");

            const string sampleSentences4 = "This is a \"test\". I said \"This is a test.\"  Any questions?";
            sents = sentDetect.SentDetect(sampleSentences4);
            Assert.AreEqual(sents.Length, 3);
            probs = sentDetect.GetSentenceProbabilities();
            Assert.AreEqual(probs.Length, 3);
            Assert.AreEqual(sents[0], "This is a \"test\".");
            Assert.AreEqual(sents[1], "I said \"This is a test.\"");
            Assert.AreEqual(sents[2], "Any questions?");

            const string sampleSentences5 = "This is a one sentence test space at the end.    ";
            sents = sentDetect.SentDetect(sampleSentences5);
            Assert.AreEqual(1, sentDetect.GetSentenceProbabilities().Length);
            Assert.AreEqual(sents[0], "This is a one sentence test space at the end.");

            const string sampleSentences6 = "This is a one sentences test with tab at the end.            ";
            sents = sentDetect.SentDetect(sampleSentences6);
            Assert.AreEqual(sents[0], "This is a one sentences test with tab at the end.");

            const string sampleSentences7 = "This is a test.    With spaces between the two sentences.";
            sents = sentDetect.SentDetect(sampleSentences7);
            Assert.AreEqual(sents[0], "This is a test.");
            Assert.AreEqual(sents[1], "With spaces between the two sentences.");

            const string sampleSentences9 = "";
            sents = sentDetect.SentDetect(sampleSentences9);
            Assert.AreEqual(0, sents.Length);

            const string sampleSentences10 = "               "; // whitespaces and tabs
            sents = sentDetect.SentDetect(sampleSentences10);
            Assert.AreEqual(0, sents.Length);

            const string sampleSentences11 = "This is test sentence without a dot at the end and spaces          ";
            sents = sentDetect.SentDetect(sampleSentences11);
            Assert.AreEqual(sents[0], "This is test sentence without a dot at the end and spaces");
            probs = sentDetect.GetSentenceProbabilities();
            Assert.AreEqual(1, probs.Length);

            const string sampleSentence12 = "    This is a test.";
            sents = sentDetect.SentDetect(sampleSentence12);
            Assert.AreEqual(sents[0], "This is a test.");

            const string sampleSentence13 = " This is a test";
            sents = sentDetect.SentDetect(sampleSentence13);
            Assert.AreEqual(sents[0], "This is a test");

            // Test that sentPosDetect also works
            var pos = sentDetect.SentPosDetect(sampleSentences2);
            Assert.AreEqual(pos.Length, 2);
            probs = sentDetect.GetSentenceProbabilities();
            Assert.AreEqual(probs.Length, 2);
            Assert.AreEqual(new Span(0, 15), pos[0]);
            Assert.AreEqual(new Span(16, 56), pos[1]);
        }


    }
}