using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Vision;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]


public class BinaryRegionsTracking
{
    public int ExpectedConnectedComponents { get; set; }
    public struct DistanceConnectedComponentCollection
    {
        public double Distance;
        public ConnectedComponentCollection ConnCompCollection; 
    }

    private static double distance (OpenCV.Net.Point2f a,OpenCV.Net.Point2f b)
    {
        return Math.Pow(b.X-a.X,2) + Math.Pow(b.Y-a.Y,2);
    }
    

	private DistanceConnectedComponentCollection permute(ConnectedComponentCollection current, 
								int l, int r, ConnectedComponentCollection previous)
	{
		if (l == r)
        {
            double totalDistance = 0;
            var newPermutationCopy = new ConnectedComponentCollection(current.ImageSize);
            for (int i = 0; i < previous.Count; i++)
            {
                if (float.IsNaN(current[i].Centroid.X))
                // maintain same state as previous frame
                {
                    //var temp = new ConnectedComponent();
                    var temp = ConnectedComponent.FromContour(previous[i].Contour);
                    temp.Centroid = new OpenCV.Net.Point2f(previous[i].Centroid.X,previous[i].Centroid.Y);
                    temp.Area = previous[i].Area;
                    temp.MajorAxisLength = previous[i].MajorAxisLength;
                    temp.MinorAxisLength = previous[i].MinorAxisLength;
                    temp.Orientation = previous[i].Orientation;
                    temp.Patch = previous[i].Patch;
                    newPermutationCopy.Add(temp);
                }
                else
                {
                    totalDistance += distance(current[i].Centroid,previous[i].Centroid);
                    newPermutationCopy.Add(current[i]);
                }
            }
            return new DistanceConnectedComponentCollection() {Distance = totalDistance, ConnCompCollection = newPermutationCopy};
        }
		else
		{ 
            var bestCandidateTillNow = new DistanceConnectedComponentCollection() {Distance = Double.MaxValue, ConnCompCollection = null};
			for (int i = l; i <= r; i++)
			{ 
				swap(current, l, i);
				
                var currentCandidate = permute(current, l + 1, r,previous);
                if (currentCandidate.Distance < bestCandidateTillNow.Distance)
                {
                   bestCandidateTillNow = currentCandidate;
                }
				swap(current, l, i);
			}
            return bestCandidateTillNow;
		} 
	}

	public static ConnectedComponentCollection swap(ConnectedComponentCollection a, 
							int i, int j)
	{ 
		ConnectedComponent temp;  
		temp = a[i] ; 
		a[i] = a[j]; 
		a[j] = temp;
		return a;
	}

// This code is contributed by mits 

    public IObservable<ConnectedComponentCollection> Process(IObservable<ConnectedComponentCollection> source)
    {
        ConnectedComponentCollection previous = null;
        return source.Select(value => 
        { 

            if (previous==null)
            {
                previous = new ConnectedComponentCollection(value,value.ImageSize);
                return previous; //bestPermute.permuted;
            }
            var VirtualConnectedComponent = new ConnectedComponent();
            VirtualConnectedComponent.Centroid = new OpenCV.Net.Point2f(float.NaN,float.NaN);

            var valueCopy = new ConnectedComponentCollection(value.ImageSize);
            for (int i = 0; i < value.Count; i++)
            {
                valueCopy.Add(value[i]);
            }
            
            // less detected individuals in this frame than expected (specified)
            // TO DO: til min(previous.count, ExpectedConnectedComponents);
            if (value.Count <  ExpectedConnectedComponents)
                 for (int i =  value.Count; i < ExpectedConnectedComponents; i++)
                 {
                    //var temp = ConnectedComponent.FromContour(value[0].Contour);
                    //temp.Centroid = new OpenCV.Net.Point2f(float.NaN,float.NaN);
                    valueCopy.Add(VirtualConnectedComponent);
                 }

            var bestPermute = permute(valueCopy,0,valueCopy.Count-1,previous);
            previous = bestPermute.ConnCompCollection;

            return bestPermute.ConnCompCollection;
        });
    }
}

            // var distanceTable = new Double[5,5];
            // for (int i = 0; i < current.Length; i++)
            // {
            //     for (int j = 0; j < previous.Length; i++) //(int j = 0 + i;
            //     {
            //         distanceTable[i,j] = distance (value[i].Centroid, previous[j].Centroid);
            //     }
            // }
