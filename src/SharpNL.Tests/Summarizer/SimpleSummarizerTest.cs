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
using SharpNL.Globalization;
using SharpNL.SentenceDetector;
using SharpNL.Summarizer;
using SharpNL.Tokenize;

namespace SharpNL.Tests.Summarizer {
    [TestFixture]
    internal class SimpleSummarizerTest {
        /// <summary>
        /// The test sentences
        /// </summary>
        /// <seealso href="http://edition.cnn.com/2015/03/15/americas/brazil-protests/index.html"/>
        private const string testSentences = @"
Demonstrators took to the streets across Brazil on Sunday, protesting corruption and demanding the impeachment of President Dilma Rousseff.
Her administration is struggling amid a weak economy and a massive corruption scandal involving the country's state-run oil company.
Demonstrators in Brazil protest against the government of President Dilma Rousseff in Paulista Avenue in Sao Paulo on March 15, 2015. 
Brazil protesters to Rousseff: Get out 13 photos
EXPAND GALLERY
""I love Brazil. I love my country. And I am tired of corruption. We are tired of corruption. It doesn't matter which political party you are from, we are tired of being robbed,"" a protester told CNN in Sao Paulo, where people packed the main Paulista Avenue.
In Rio de Janeiro, they gathered along Copacabana beach, while in the capital, Brasilia, protesters marched on government headquarters.
The mood was festive. Many demonstrators wore the country's colors -- green, blue and yellow -- waved flags, and chanted: ""Out Dilma.""
Amid complaints about the economy, protesters say they are incensed because Brazilian investigators are unraveling a huge money-laundering and bribery case centered around Petrobras, the country's national oil company. Dozens of politicians, some in Rousseff's party, are accused of accepting millions in payments.
The President has not been implicated in the investigation, but she was the Energy Minister and chairwoman of Petrobras during much of the time that the alleged corruption took place.
Why are protesters furious with Brazil's President?
Sunday night Rousseff sent two of her ministers to address the nation at a televised press conference. The justice minister and the general secretary said that the government was listening and would announce changes in several days designed to combat corruption.
The announcement did little to quiet the protests in some cities. Many protesters banged pots and pans and honked car horns.
";

        [Test]
        public void TestSummarizer() {

            var sum = new SimpleSummarizer(Culture.GetCulture("en"), SimpleSummarizerMethods.FrequentWords) {
                NumberOfSentences = 1
            };

            string text;

            using (var sentFile = Tests.OpenFile("/opennlp/models/en-sent.bin")) {
                using (var sentToken = Tests.OpenFile("/opennlp/models/en-token.bin")) {
                    text = sum.Summarize(testSentences, 
                        new SentenceDetectorME(new SentenceModel(sentFile)), 
                        new TokenizerME(new TokenizerModel(sentToken)));

                }
            }

            Assert.IsNotNullOrEmpty(text);
            Assert.AreEqual("Amid complaints about the economy, protesters say they are incensed because Brazilian investigators are unraveling a huge money-laundering and bribery case centered around Petrobras, the country's national oil company.", text);
        }

        
    }
}