#SharpNL

## What is this?

> An awesome and independent reimplementation of the [Apache OpenNLP] software library in C#

> [![Build status](https://ci.appveyor.com/api/projects/status/r11n96yn48jpt6v4/branch/development?svg=true)](https://ci.appveyor.com/project/knuppe/sharpnl/branch/development)
> [![Coverity Scan](https://scan.coverity.com/projects/5813/badge.svg)](https://scan.coverity.com/projects/5813) 

## Release

> Public release available at [NuGet] \(Current version: 1.1)

## Main features/characteristics

> - Fully C# managed NET 4.5 library.
> - Fully compatible with the OpenNLP models (1.5, 1.5.3 and 1.6).
> - Was built from scratch by hand, without any assist tool in order to maximize the synergy with .net technology.
> - There are [analyzers](https://github.com/knuppe/SharpNL/wiki/Analyzers) that help a lot the implementation and abstraction of this library.  
> - The heavy operations (like training) can be monitored and cancelled.
> - Some file formats were revamped/improved (Ad and Penn Treebank).
> - The deprecated methods from OpenNLP were not ported!
> - English inflection tools.
> - WordNet 3.0 integration.

## Goals

> Implement the "best" library of natural language processing in C#, which means:
> - Be as lightweight as possible
> - Have a good set of tools available in a single library

## TODO

> - Ensure that the library is compatible with Mono.

> [How to contribute](contributing.md)

## WIP/Planned
> - [Vector Classifier](https://en.wikipedia.org/wiki/Support_vector_machine)
> - Run a Profiler and improve the code to reduce memory and CPU utilization.

## Support

As a human being I like to be honest, I believe that someday our kind will transcend money... 
But unfortunately, while we have not reached this day I need some money to make my living. 
If you like this project or need to use this library in the future please consider making a 
donation (anything helps), writing this library takes a HUGE amount of my time, effort and 
resources. But I do it because I'm passionate about it, and hope to make an impact on the 
world while also sharing my little knowledge as a human being.

Please, consider donating as a thank you.

[![donate](resources/donate.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=7SWNPAPJNSARC)

[![bean](resources/bean.gif)](#)

[NuGet]: https://www.nuget.org/packages/Knuppe.SharpNL/
[Apache OpenNLP]: http://opennlp.apache.org
[CoGrOO]: http://cogroo.sourceforge.net/
