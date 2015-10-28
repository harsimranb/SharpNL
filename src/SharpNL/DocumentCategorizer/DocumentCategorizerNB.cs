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

using System;
using System.Collections.Generic;
using SharpNL.ML;
using SharpNL.ML.NaiveBayes;
using SharpNL.Utility;

namespace SharpNL.DocumentCategorizer {
    /// <summary>
    /// Represents a Naive Bayes document categorizer.
    /// </summary>    
    public class DocumentCategorizerNB : IDocumentCategorizer {

        private static IFeatureGenerator defaultFeatureGenerator = new BagOfWordsFeatureGenerator();

        private DocumentCategorizerModel model;
        private DocumentCategorizerContextGenerator contextGenerator;

        #region + Constructors .

        /// <summary>
        /// Initializes a new instance of the <see cref="DocumentCategorizerNB"/> class. The default feature generation is used.
        /// </summary>
        /// <param name="model">The model.</param>
        public DocumentCategorizerNB(DocumentCategorizerModel model) {
            if (model == null)
                throw new ArgumentNullException("model");

            this.model = model;
            contextGenerator = new DocumentCategorizerContextGenerator(model.Factory.FeatureGenerators);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DocumentCategorizerNB"/> class with a doccat model and custom feature
        /// generation. The feature generation must be identical to the configuration at training time.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="featureGenerators">The feature generators.</param>
        /// <exception cref="ArgumentNullException">model</exception>
        public DocumentCategorizerNB(DocumentCategorizerModel model, params IFeatureGenerator[] featureGenerators) {
            if (model == null)
                throw new ArgumentNullException("model");

            this.model = model;
            contextGenerator = new DocumentCategorizerContextGenerator(featureGenerators);
        }

        #endregion


        /// <summary>
        /// Categorizes the specified text.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <returns>An array of the probabilities for each of the different outcomes, all of which sum to 1.</returns>
        public double[] Categorize(string[] tokens) {
            return Categorize(tokens, new Dictionary<string, object>());
        }

        /// <summary>
        /// Categorizes the specified document.
        /// </summary>
        /// <param name="document">The document string.</param>
        /// <returns>An array of the probabilities for each of the different outcomes, all of which sum to 1.</returns>
        public double[] Categorize(string document) {
            var tokenizer = model.Factory.Tokenizer;
            return Categorize(tokenizer.Tokenize(document), new Dictionary<string, object>());
        }

        /// <summary>
        /// Categorizes the specified text with extra informations.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="extraInformation">The extra information.</param>
        /// <returns>An array of the probabilities for each of the different outcomes, all of which sum to 1.</returns>
        public double[] Categorize(string[] tokens, Dictionary<string, object> extraInformation) {
            return model.MaxentModel.Eval(contextGenerator.GetContext(tokens, extraInformation));
        }

        /// <summary>
        /// Categorizes the specified document with extra information.
        /// </summary>
        /// <param name="document">The document.</param>
        /// <param name="extraInformation">The extra information.</param>
        /// <returns>An array of the probabilities for each of the different outcomes, all of which sum to 1.</returns>
        public double[] Categorize(string document, Dictionary<string, object> extraInformation) {
            var tokenizer = model.Factory.Tokenizer;
            return Categorize(tokenizer.Tokenize(document), extraInformation);
        }

        /// <summary>
        /// Returns the best category for the given outcome.
        /// </summary>
        /// <param name="outcome">The outcome.</param>
        /// <returns>The best category.</returns>
        public string GetBestCategory(double[] outcome) {
            return model.MaxentModel.GetBestOutcome(outcome);
        }

        /// <summary>
        /// Returns the category index.
        /// </summary>
        /// <param name="category">The category.</param>
        /// <returns>Category index.</returns>
        public int GetIndex(string category) {
            return model.MaxentModel.GetIndex(category);
        }

        public string GetCategory(int index) {
            return model.MaxentModel.GetOutcome(index);
        }

        /// <summary>
        /// Gets the number of categories.
        /// </summary>
        /// <value>The number of categories.</value>
        public int NumberOfCategories {
            get { return model.MaxentModel.GetNumOutcomes(); }
        }

        /// <summary>
        /// Returns a map in which the key is the category name and the value is the score.
        /// </summary>
        /// <param name="text">text the input text to classify.</param>
        /// <returns>The dictionary with the categories with the scores.</returns>
        public Dictionary<string, double> ScoreMap(string text) {
            var map = new Dictionary<string, double>();
            var cats = Categorize(text);
            var size = NumberOfCategories;
            for (var i = 0; i < size; i++) {
                var cat = GetCategory(i);
                map[cat] = cats[GetIndex(cat)];
            }
            return map;
        }

        /// <summary>
        /// Returns a map in which the key is the category name and the value is the score.
        /// </summary>
        /// <param name="tokens">The sentence tokens to classify.</param>
        /// <returns>The dictionary with the categories with the scores.</returns>
        public Dictionary<string, double> ScoreMap(string[] tokens) {
            var map = new Dictionary<string, double>();
            var cats = Categorize(tokens);
            var size = NumberOfCategories;
            for (var i = 0; i < size; i++) {
                var cat = GetCategory(i);
                map[cat] = cats[GetIndex(cat)];
            }
            return map;
        }

        /// <summary>
        /// Returns a map with the score as a key in ascending order.
        /// </summary>
        /// <param name="text">Text the input text to classify.</param>
        /// <returns>A dictionary of categories with the score.</returns>
        /// <returns>
        /// Many categories can have the same score, hence the set as value
        /// </returns>
        public SortedDictionary<double, List<string>> SortedScoreMap(string text) {
            var descendingMap = new SortedDictionary<double, List<string>>();
            var categorized = Categorize(text);
            var catSize = NumberOfCategories;
            for (var i = 0; i < catSize; i++) {
                var category = GetCategory(i);
                var score = categorized[GetIndex(category)];
                if (descendingMap.ContainsKey(score)) {
                    descendingMap[score].Add(category);
                } else {
                    descendingMap[score] = new List<string> {category};
                }
            }
            return descendingMap;
        }



        public static DocumentCategorizerModel Train(
            string languageCode,
            IObjectStream<DocumentSample> samples,
            TrainingParameters parameters,
            params IFeatureGenerator[] featureGenerators) {

            var manifestInfoEntries = new Dictionary<string, string>();

            parameters.Set(Parameters.Algorithm, Parameters.Algorithms.NaiveBayes);

            var model = GetTrainedInnerModel(samples, parameters, manifestInfoEntries, featureGenerators);

            return new DocumentCategorizerModel(languageCode, model, manifestInfoEntries, new DocumentCategorizerFactory());
        }

        public static DocumentCategorizerModel Train(
            string languageCode,
            IObjectStream<DocumentSample> samples,
            TrainingParameters parameters,
            DocumentCategorizerFactory factory) {

            var manifestInfoEntries = new Dictionary<string,string>();

            parameters.Set(Parameters.Algorithm, Parameters.Algorithms.NaiveBayes);

            var model = GetTrainedInnerModel(samples, parameters, manifestInfoEntries, factory.FeatureGenerators);

            return new DocumentCategorizerModel(languageCode, model, manifestInfoEntries, factory);
        }


        protected static NaiveBayesModel GetTrainedInnerModel(
            IObjectStream<DocumentSample> samples,
            TrainingParameters parameters,
            Dictionary<string, string> manifestInfoEntries,
            params IFeatureGenerator[] featureGenerators) {

            var trainer = TrainerFactory.GetEventTrainer(parameters, manifestInfoEntries, null);
            if (trainer == null)
                throw new InvalidOperationException("The event trainer is not supported.");


            var model = trainer.Train(new DocumentCategorizerEventStream(samples, featureGenerators));

            return model as NaiveBayesModel;
        }
    }
}