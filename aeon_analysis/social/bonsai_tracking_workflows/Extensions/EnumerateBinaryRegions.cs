using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Vision;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class EnumerateBinaryRegions
{
    public IObservable<ConnectedComponentCollection> Process(IObservable<ConnectedComponentCollection> source)
    {
        return source;
    }
}
