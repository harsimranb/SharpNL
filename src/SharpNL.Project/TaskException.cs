﻿// 
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

namespace SharpNL.Project {
    /// <summary>
    /// Represents a task exception.
    /// </summary>
    public class TaskException : Exception {
        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Exception"/> class with a specified error message.
        /// </summary>
        /// <param name="projectTask">The project task.</param>
        /// <param name="message">The message that describes the error. </param>
        public TaskException(ProjectTask projectTask, string message) : base(message) {
            ProjectTask = projectTask;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TaskException"/> class.
        /// </summary>
        /// <param name="projectTask">The project task.</param>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The inner exception.</param>
        internal TaskException(ProjectTask projectTask, string message, Exception innerException) : base(message, innerException) {
            ProjectTask = projectTask;
        }

        /// <summary>
        /// Gets the task.
        /// </summary>
        /// <value>The task.</value>
        public ProjectTask ProjectTask { get; private set; }
    }
}