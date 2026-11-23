using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNCProduction
{
    public class CHole
    {
        public float holeCenterMmX = 0.0f;
        public float holeCenterMmY = 0.0f;
        public float diameterMm = 0.0f;

        public CHole(float holeCenterX, float holeCenterY, float diameter)
        {
            this.holeCenterMmX = holeCenterX;
            this.holeCenterMmY = holeCenterY;
            this.diameterMm = diameter;
        }
    }
}
