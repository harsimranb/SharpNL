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
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace SharpNL.Utility.FeatureGen {
    /// <summary>
    /// Word2vec and clark clustering style lexicon dictionary.
    /// </summary>
    [TypeClass("opennlp.tools.util.featuregen.W2VClassesDictionary")]
    public class W2VClassesDictionary {

        private readonly Dictionary<string, string> tokenToClusterMap;

        /// <summary>
        /// Initializes a new instance of the <see cref="W2VClassesDictionary"/> class.
        /// </summary>
        /// <param name="inputStream">The input stream.</param>
        public W2VClassesDictionary(Stream inputStream) {
            tokenToClusterMap = new Dictionary<string, string>();

            var reader = new StreamReader(inputStream, Encoding.UTF8);

            string line;
            while ((line = reader.ReadLine()) != null) {
                var parts = line.Split(' ');

                if (parts.Length == 2 || parts.Length == 3) {
                    tokenToClusterMap.Add(parts[0], parts[1]);
                }
            }
        }


        /// <summary>
        /// Lookups the token.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>System.String.</returns>
        public string LookupToken(string value) {
            return tokenToClusterMap[value];
        }

        /// <summary>
        /// Gets the <see cref="System.String"/> with the specified key.
        /// </summary>
        /// <param name="key">The key to look-up.</param>
        /// <returns>The brown class if such key is in the dictionary.</returns>
        public string this[string key] {
            get { return tokenToClusterMap[key]; }
        }

        internal static void Serialize(object artifact, Stream outputStream) {

            var w2v = artifact as W2VClassesDictionary;
            if (w2v == null)
                throw new InvalidOperationException();

            using (var writer = new StreamWriter(outputStream, Encoding.UTF8, 1024, true)) {
                foreach (var pair in w2v.tokenToClusterMap) {
                    writer.WriteLine("{0}\t{1}\n", pair.Key, pair.Value);
                }
                writer.Flush();
            }
        }

        internal static object Deserialize(Stream inputStream) {
            return new W2VClassesDictionary(inputStream);
        }


        
    }
}