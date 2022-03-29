using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class CumulativeBuffer
{
    public int Count { get; set; }

    public IObservable<TSource[]> Process<TSource>(IObservable<TSource> source)
    {
        return Observable.Defer(() =>
        {
            var capacity = Count;
            var buffer = new Queue<TSource>(capacity);
            return source.Select(value =>
            {
                while (buffer.Count >= capacity)
                {
                    buffer.Dequeue();
                }
                buffer.Enqueue(value);
                return buffer.ToArray();
            });
        });
    }
}
