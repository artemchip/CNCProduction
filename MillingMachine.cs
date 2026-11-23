using DirectShowLib;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using static Emgu.CV.DISOpticalFlow;

namespace CNCProduction
{
    public class MillingMachine : CNCGeneralMachine
    {
        // Status
        private bool transformPending = false;
        private int transformProgressPercent = 0;
        private List<ALVertice> alVertices = new List<ALVertice>();
        private List<CHole> holesToDrill = new List<CHole>();
        private List<PointF[]> contoursToCutMm = new List<PointF[]>();

        // Отрисовка превью
        private Pen contourPen = new Pen(new SolidBrush(Color.Green));
        private Brush drilledBrush = new SolidBrush(Color.Green);

        // Константы для CNC 3018
        private int drillingRpm = 4000;
        private int zSinkRateBigMmPerMin = 80;
        private int zRetractRateBigMmPerMin = 150;
        private int zSinkRateSmallMmPerMin = 120;
        private int zRetractRateSmallMmPerMin = 200;
        private int cuttingRpm = 7000;
        private int cuttingRateMmPerMin = 300;
        private float iCutMm = 40.0f;
        private float iGapMm = 1.5f;

        // Мои константы
        private float cuttingToolDiameterMm = 1.5f;
        private float toolChangeZMm = 38.0f;
        private float probeZMm = 18.0f;
        private float epDepthMm = 3.5f;
        private float layerHeightMm = 0.1f;

        public MillingMachine(String configFilePath, int xCamerasNeeded, int xPreviewBoxWidth, int xPreviewBoxHeight, EventHandler xStateChanged, EventHandler<ErrorEventArgs> xErrorOccurred, EventHandler<NewCameraPreviewImageArgs> xNewCameraPreviewImage, EventHandler<UserInputRequiredArgs> xUserInputRequired) : base(configFilePath, xCamerasNeeded, xPreviewBoxWidth, xPreviewBoxHeight, xStateChanged, xErrorOccurred, xNewCameraPreviewImage, xUserInputRequired)
        {
            // Константа
            calibZMm = 18.0f;
            probeZMm = calibZMm + 0.0f;
            machineIP = "";
        }

        public List<PointF[]> SplitContour(PointF[] contour)
        {
            List<PointF[]> result = new List<PointF[]>();
            bool isGap = false;
            float distanceMoved = 0.0f;
            PointF? machinePosition = null;
            List<PointF> curSubcontour = new List<PointF>();
            foreach (PointF curPoint in contour)
            {
                curSubcontour.Add(curPoint);
                if (machinePosition != null)
                {
                    distanceMoved += Vector2.Distance(curPoint.ToVector2(), machinePosition?.ToVector2() ?? new Vector2(0f, 0f));
                    if (isGap && distanceMoved >= iGapMm)
                    {
                        isGap = false;
                        distanceMoved -= iGapMm;
                        curSubcontour.Clear();
                    }
                    else if (!isGap && distanceMoved >= iCutMm)
                    {
                        isGap = true;
                        distanceMoved -= iCutMm;
                        if (curSubcontour.Count > 0)
                        {
                            result.Add(curSubcontour.ToArray());
                        }
                        curSubcontour.Clear();
                    }
                }
                machinePosition = curPoint;
            }
            if (curSubcontour.Count > 0)
            {
                result.Add(curSubcontour.ToArray());
            }
            return result;
        }

        public void CalculateContoursByFileAndDPI(String imgFile, int rDPI)
        {
            // Очистить списки
            holesToDrill.Clear();
            contoursToCutMm.Clear();

            // Vectorize it
            List<PointF[]> contoursTmp = new List<PointF[]>();
            using (Bitmap bmp = new Bitmap(imgFile))
            {
                // Обработка
                float pixelsPerMm = (((float)rDPI) / 25.4f);
                int widthShouldBePx = (int)(pixelsPerMm * maxTravelMmX);
                int heightShouldBePx = (int)(pixelsPerMm * maxTravelMmY);
                if (bmp.Width != widthShouldBePx || bmp.Height != heightShouldBePx)
                {
                    throw new Exception("Разрешение согласно DPI должно быть " + widthShouldBePx.ToString() + " x " + heightShouldBePx.ToString());
                }
                Mat binary = new Mat();
                Mat grey = new Mat();
                CvInvoke.CvtColor(bmp.ToMat(), grey, ColorConversion.Rgb2Gray);
                CvInvoke.Threshold(grey, binary, 85, 255, ThresholdType.Binary);
                VectorOfVectorOfPoint contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
                Mat hier = new Mat();
                CvInvoke.FindContours(binary, contours, hier, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);
                float maxPerimeterMm = (maxTravelMmX + maxTravelMmY) * 2f;
                for (int i = 0; i < contours.Size; i++)
                {
                    // Размер
                    double perimeterPx = CvInvoke.ArcLength(contours[i], true);
                    if (perimeterPx < 1f || perimeterPx > (0.9f * maxPerimeterMm * pixelsPerMm))
                    {
                        continue;
                    }

                    // Approx
                    VectorOfPoint approx = new VectorOfPoint();
                    CvInvoke.ApproxPolyDP(contours[i], approx, rDPI * 0.005f, false);
                    if (approx.Size > 1 && approx.Size < 100)
                    {
                        // Area
                        float areaPxSq = (float)(CvInvoke.ContourArea(approx));

                        // Draw
                        List<PointF> thisContour = new List<PointF>();
                        for (int j = 0; j < approx.Size; j++)
                        {
                            thisContour.Add(new PointF(((float)approx[j].X) / pixelsPerMm, ((float)approx[j].Y) / pixelsPerMm));
                        }

                        // First - closed
                        if (i == 0 && thisContour.Count > 2)
                        {
                            thisContour.Add(thisContour[0]);
                        }

                        // Contour size?
                        float minContourX = 100000.0f;
                        float minContourY = 100000.0f;
                        float maxContourX = 0.0f;
                        float maxContourY = 0.0f;
                        foreach (PointF point in thisContour)
                        {
                            if (point.X < minContourX)
                            {
                                minContourX = point.X;
                            }
                            if (point.Y < minContourY)
                            {
                                minContourY = point.Y;
                            }
                            if (point.X > maxContourX)
                            {
                                maxContourX = point.X;
                            }
                            if (point.Y > maxContourY)
                            {
                                maxContourY = point.Y;
                            }
                        }
                        RectangleF contourSize = new RectangleF(minContourX, minContourY, (maxContourX - minContourX), (maxContourY - minContourY));

                        // Scale
                        PointF centerPoint = new PointF(contourSize.Left + (contourSize.Width / 2.0f), contourSize.Top + (contourSize.Height / 2.0f));
                        if (contourSize.Width < (1.5f * cuttingToolDiameterMm) || contourSize.Height < (1.5f * cuttingToolDiameterMm))
                        {
                            continue;
                        } else if (contourSize.Width < (10f * cuttingToolDiameterMm) || contourSize.Height < (10f * cuttingToolDiameterMm))
                        {
                            float coeff = contourSize.Width / contourSize.Height;
                            if (coeff < 1.1f && coeff > 0.9f)
                            {
                                float circularity = (float)((4f * Math.PI * areaPxSq) / (perimeterPx * perimeterPx));
                                if (circularity > 0.8f)
                                {
                                    // This is a hole
                                    holesToDrill.Add(new CHole(centerPoint.X, centerPoint.Y, ((maxContourX - minContourX) + (maxContourY - minContourY)) / 2.0f));
                                    continue;
                                }
                            }
                        }
                        float scaleFactorX = 0.0f;
                        float scaleFactorY = 0.0f;
                        float delta = (i == 0) ? cuttingToolDiameterMm : -cuttingToolDiameterMm;
                        scaleFactorX = (contourSize.Width + delta) / contourSize.Width;
                        scaleFactorY = (contourSize.Height + delta) / contourSize.Height;
                        if (scaleFactorX < 0.5f && scaleFactorY < 0.5f)
                        {
                            if (contourSize.Width >= (1.5f * cuttingToolDiameterMm) && contourSize.Height >= (1.5f * cuttingToolDiameterMm))
                            {
                                // This is a hole
                                holesToDrill.Add(new CHole(centerPoint.X, centerPoint.Y, ((maxContourX - minContourX) + (maxContourY - minContourY)) / 2.0f));
                                continue;
                            }
                            else
                            {
                                continue;
                            }
                        }
                        else
                        {
                            // Fixup
                            PointF[] fixedContour = thisContour.ToArray();
                            for (int j = 0; j < fixedContour.Length; j++)
                            {
                                fixedContour[j].X = centerPoint.X + (scaleFactorX * (thisContour[j].X - centerPoint.X));
                                fixedContour[j].Y = centerPoint.Y + (scaleFactorY * (thisContour[j].Y - centerPoint.Y));
                            }

                            // Interpolate
                            List<PointF> newContour = new List<PointF>();
                            PointF? machinePosition = null;
                            foreach (PointF curPoint in fixedContour)
                            {
                                if ((machinePosition != null) && Vector2.Distance(curPoint.ToVector2(), (machinePosition?.ToVector2() ?? new Vector2(0f, 0f))) > 1f)
                                {
                                    // Divide by pieces
                                    float distTotal = Vector2.Distance(machinePosition?.ToVector2() ?? new Vector2(0f, 0f), curPoint.ToVector2());
                                    float distPassed = 0.0f;
                                    while (distPassed < distTotal)
                                    {
                                        distPassed += 1f;
                                        if (distPassed > distTotal) distPassed = distTotal;
                                        Vector2 interPoint = Vector2.Lerp(machinePosition?.ToVector2() ?? new Vector2(0f, 0f), curPoint.ToVector2(), (distPassed / distTotal));
                                        newContour.Add(new PointF(interPoint.X, interPoint.Y));
                                    }
                                }
                                else
                                {
                                    // Cut
                                    newContour.Add(curPoint);
                                }
                                machinePosition = curPoint;
                            }

                            // Split
                            contoursTmp.AddRange(SplitContour(newContour.ToArray()));
                            continue;
                        }

                        // We should not be here
                        throw new Exception("Assertion!");
                    }
                }
            }

            // Добавить все контуры
            contoursToCutMm.AddRange(Utils.SortContours(contoursTmp));
        }

        public int GetTransformProgressPercent()
        {
            return Math.Clamp(transformProgressPercent, 0, 100);
        }

        public Bitmap GetPreviewBitmap(int rDPI)
        {
            float pixelsPerMm = (((float)rDPI) / 25.4f);
            Bitmap bmpResult = new Bitmap((int)(maxTravelMmX * pixelsPerMm), (int)(maxTravelMmY * pixelsPerMm), System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(bmpResult))
            {
                contourPen.Color = Color.White;
                contourPen.Width = pixelsPerMm * cuttingToolDiameterMm;
                contourPen.SetLineCap(System.Drawing.Drawing2D.LineCap.Round, System.Drawing.Drawing2D.LineCap.Round, System.Drawing.Drawing2D.DashCap.Round);
                g.Clear(Color.Black);
                
                // Contours
                if (contoursToCutMm.Count > 0)
                {
                    for (int i = 0; i < contoursToCutMm.Count; i++)
                    {
                        List<PointF> gPoints = new List<PointF>();
                        PointF prevPoint = new PointF(-1f, -1f);
                        if (contoursToCutMm[i].Length == 1)
                        {
                            Rectangle holeRect = new Rectangle(
                                (int)((contoursToCutMm[i][0].X * pixelsPerMm) - ((cuttingToolDiameterMm * pixelsPerMm) / 2f)),
                                (int)((contoursToCutMm[i][0].Y * pixelsPerMm) - ((cuttingToolDiameterMm * pixelsPerMm) / 2f)),
                                (int)(cuttingToolDiameterMm * pixelsPerMm),
                                (int)(cuttingToolDiameterMm * pixelsPerMm)
                            );
                            g.FillEllipse(Brushes.Yellow, holeRect);
                        }
                        else
                        {
                            for (int j = 0; j < contoursToCutMm[i].Length; j++)
                            {
                                PointF curPoint = new PointF(contoursToCutMm[i][j].X * pixelsPerMm, contoursToCutMm[i][j].Y * pixelsPerMm);
                                if (prevPoint.X != -1f && prevPoint.Y != -1f)
                                {
                                    g.DrawLine(contourPen, prevPoint, curPoint);
                                }
                                prevPoint = curPoint;
                            }
                        }
                    }
                }

                // Holes
                if (holesToDrill.Count > 0)
                {
                    for (int i = 0; i < holesToDrill.Count; i++)
                    {
                        Rectangle holeRect = new Rectangle(
                            (int)((holesToDrill[i].holeCenterMmX * pixelsPerMm) - ((holesToDrill[i].diameterMm * pixelsPerMm) / 2f)),
                            (int)((holesToDrill[i].holeCenterMmY * pixelsPerMm) - ((holesToDrill[i].diameterMm * pixelsPerMm) / 2f)),
                            (int)(holesToDrill[i].diameterMm * pixelsPerMm),
                            (int)(holesToDrill[i].diameterMm * pixelsPerMm)
                        );
                        g.FillEllipse(drilledBrush, holeRect);
                    }
                }
            }
            return bmpResult;
        }

        public bool CanAdjustSpindleSpeed()
        {
            return !spindleActive;
        }

        public float GetTransformTimeSeconds()
        {
            // Transforming
            float totalSlowDistMm = 0.0f;
            float totalFastDistMm = 0.0f;
            float vmX = 0.0f;
            float vmY = 0.0f;
            foreach (PointF[] contour in contoursToCutMm)
            {
                bool firstDone = false;
                foreach (PointF p in contour)
                {
                    float curMoveDist = (float)Math.Sqrt(Math.Pow(p.X - vmX, 2) + Math.Pow(p.Y - vmY, 2));
                    if (!firstDone)
                    {
                        totalFastDistMm += curMoveDist;
                        firstDone = true;
                    }
                    else
                    {
                        totalSlowDistMm += curMoveDist;
                    }
                    vmX = p.X;
                    vmY = p.Y;
                }
            }
            float procRateMmPerMin = cuttingRateMmPerMin;
            int layerCount = (int)Math.Ceiling(epDepthMm / layerHeightMm);
            float transformTimeSeconds = (((totalSlowDistMm * layerCount) / procRateMmPerMin) + ((totalFastDistMm * layerCount) / Math.Min(maxFeedRateMmPerMinX, maxFeedRateMmPerMinY))) * 60.0f;

            // Drilling
            List<List<CHole>> ourHolesByTool = GroupHolesAndSortByMoveDistanceAndDiameter();
            float drillTimeSeconds = 0.0f;

            // Drilling
            float maX = 0.0f;
            float maY = 0.0f;
            for (int i = 0; i < ourHolesByTool.Count; i++)
            {
                // 1 min to change tool
                drillTimeSeconds += 60.0f;

                // Each Hole
                for (int j = 0; j < ourHolesByTool[i].Count; j++)
                {
                    // Move
                    CHole thisHole = ourHolesByTool[i][j];
                    drillTimeSeconds += ((float)MillisecToExecuteMove(thisHole.holeCenterMmX - maX, thisHole.holeCenterMmY - maY, 0f, maxFeedRateMmPerMinX, maxFeedRateMmPerMinY, maxFeedRateMmPerMinZ)) / 1000.0f;
                    maX = thisHole.holeCenterMmX;
                    maY = thisHole.holeCenterMmY;

                    // Drill
                    if (thisHole.diameterMm >= 0.5f)
                    {
                        drillTimeSeconds += ((float)MillisecToExecuteMove(0f, 0f, -epDepthMm, maxFeedRateMmPerMinX, maxFeedRateMmPerMinY, zSinkRateBigMmPerMin)) / 1000.0f;
                        drillTimeSeconds += ((float)MillisecToExecuteMove(0f, 0f, epDepthMm, maxFeedRateMmPerMinX, maxFeedRateMmPerMinY, zRetractRateBigMmPerMin)) / 1000.0f;
                    }
                    else
                    {
                        drillTimeSeconds += ((float)MillisecToExecuteMove(0f, 0f, -epDepthMm, maxFeedRateMmPerMinX, maxFeedRateMmPerMinY, zSinkRateSmallMmPerMin)) / 1000.0f;
                        drillTimeSeconds += ((float)MillisecToExecuteMove(0f, 0f, epDepthMm, maxFeedRateMmPerMinX, maxFeedRateMmPerMinY, zRetractRateSmallMmPerMin)) / 1000.0f;
                    }
                }
            }

            // Sum
            return drillTimeSeconds + transformTimeSeconds;
        }

        private List<List<CHole>> GroupHolesAndSortByMoveDistanceAndDiameter()
        {
            // Group And Sort
            List<CHole> toolList = holesToDrill.DistinctBy((e) => e.diameterMm).OrderByDescending((e) => e.diameterMm).ToList();
            List<List<CHole>> results = new List<List<CHole>>();
            for (int i = 0; i < toolList.Count; i++)
            {
                List<CHole> sortedHoleList = Utils.SortHoles(holesToDrill.Where((e) => e.diameterMm == toolList[i].diameterMm).ToList());
                results.Add(sortedHoleList);
            }

            // Return
            return results;
        }

        private void DoTransformProcess()
        {
            try
            {
                // Reset
                shouldCancelOp = false;
                transformProgressPercent = 0;
                int drillingProgressPercent = 0;
                int cuttingProgressPercent = 0;
                transformPending = true;
                SetBusyWithLongTask(true);
                StateChanged.Invoke(this, new EventArgs());

                // Проверка калибровки
                if (GetCamToToolDistX() == 0.0f || GetCamToToolDistY() == 0.0f)
                {
                    throw new Exception("Смещение камера-инструмент не откалибровано.");
                }
                if (contoursToCutMm.Count <= 0 && holesToDrill.Count <= 0)
                {
                    throw new Exception("Нет посчитанных контуров и/или отверстий для выполнения обработки.");
                }

                // Layer
                int layerCount = (int)Math.Ceiling(epDepthMm / layerHeightMm);
                if (layerCount <= 0)
                {
                    throw new Exception("Слишком малая глубина обработки (меньше одного слоя).");
                }

                // Перемещаемся в центр панели
                MoveToAbsolutePositionAndWaitForCompletion(false, maxTravelMmX / 2.0f, maxTravelMmY / 2.0f, probeZMm, -1.0f);

                // Ставим Z
                int res = AskUserForInput("Переместитесь по Z так, чтобы сверло/фреза отступала от заготовки примерно на 1 мм. Это будет безопасная высота. Рабочая будет на " + epDepthMm.ToString("0.0") + " мм ниже (константа). Убедитесь что в константах в коде указана правильная высота слоя и скорости подачи.", new List<string> { "Готово", "Отмена" }, 2, false, true, false);
                if (res > 0)
                {
                    throw new Exception("Операция отменена.");
                }
                float safeZMm = GetAbsZMm();
                float workZMm = GetAbsZMm() - epDepthMm;
                int i = 0;

                // Drill
                List<List<CHole>> ourHolesByTool = GroupHolesAndSortByMoveDistanceAndDiameter();
                int holesDrilled = 0;
                int holesTotalThisRun = ourHolesByTool.Sum(o => o.Count);
                i = 0;
                while (i < ourHolesByTool.Count)
                {
                    // Ask To Change Tool
                    MoveToAbsolutePositionAndWaitForCompletion(false, null, null, toolChangeZMm, -1.0f);
                    if (i > 0)
                    {
                        res = AskUserForInput("Если все прошло нормально, поставьте в патрон сверло диаметром " + ourHolesByTool[i].First().diameterMm.ToString("0.0") + " мм и нажмите [готово]. Если предыдущее (" + ourHolesByTool[i - 1].First().diameterMm.ToString("0.0") + " мм) сломалось - поставьте его же новую версию и нажмите [повтор].", new List<string> { "Готово", "Повтор", "Отмена" }, 2, false, false, false);
                        if (res > 1)
                        {
                            throw new Exception("Операция отменена.");
                        }
                        else if (res > 0)
                        {
                            i--;
                        }
                    }
                    else
                    {
                        res = AskUserForInput("Поставьте в патрон сверло диаметром " + ourHolesByTool[i].First().diameterMm.ToString("0.0") + " мм и нажмите [готово].", new List<string> { "Готово", "Отмена" }, 2, false, false, false);
                        if (res > 0)
                        {
                            throw new Exception("Операция отменена.");
                        }
                    }

                    // Tool Installed, Move To More Working Height
                    MoveToAbsolutePositionAndWaitForCompletion(false, null, null, probeZMm, -1.0f);

                    // Start spindle
                    SetSpindleRpm(drillingRpm, true);

                    // Drill each of holes
                    for (int j = 0; j < ourHolesByTool[i].Count; j++)
                    {
                        // Отверстие
                        CHole thisHole = ourHolesByTool[i][j];

                        // Переместиться
                        Vector2 tgtPos = new Vector2(thisHole.holeCenterMmX, thisHole.holeCenterMmY);
                        MoveToAbsolutePositionAndWaitForCompletion(false, tgtPos.X, tgtPos.Y, safeZMm, -1.0f);

                        // Сверло вниз
                        MoveToAbsolutePositionAndWaitForCompletion(true, null, null, workZMm, (thisHole.diameterMm < 0.5f) ? zSinkRateSmallMmPerMin : zSinkRateBigMmPerMin);

                        // Сверло вверх
                        MoveToAbsolutePositionAndWaitForCompletion(true, null, null, safeZMm, (thisHole.diameterMm < 0.5f) ? zRetractRateSmallMmPerMin : zRetractRateBigMmPerMin);

                        // Прогресс
                        holesDrilled++;
                        drillingProgressPercent = (int)((((float)holesDrilled) * 100.0f) / ((float)holesTotalThisRun));
                        cuttingProgressPercent = 0;
                        transformProgressPercent = (int)Math.Clamp((drillingProgressPercent * 0.5f) + (cuttingProgressPercent * 0.5f), 0f, 100f);
                        StateChanged.Invoke(this, new EventArgs());

                        // Отмена
                        if (shouldCancelOp)
                        {
                            break;
                        }
                    }

                    // Stop spindle
                    SetSpindleRpm(0, true);

                    // Next Tool
                    i++;

                    // Move To Probe Height
                    MoveToAbsolutePositionAndWaitForCompletion(false, null, null, probeZMm, -1.0f);

                    // Ask If Last Tool Broken
                    if (i >= (ourHolesByTool.Count - 1))
                    {
                        res = AskUserForInput("Если это сверло (" + ourHolesByTool[i].First().diameterMm.ToString("0.0") + " мм) сломалось, замените его и нажмите [повтор], в противном случае нажмите [все нормально].", new List<string> { "Все нормально", "Повтор" }, 2, false, false, false);
                        if (res > 0)
                        {
                            i--;
                        }
                    }

                    // Отмена
                    if (shouldCancelOp)
                    {
                        throw new Exception("Операция отменена.");
                    }
                }

                // Останавливаем шпиндель
                SetSpindleRpm(0, true);

                // Контуры
                if (contoursToCutMm.Count > 0)
                {
                    // Установить фрезу
                    MoveToAbsolutePositionAndWaitForCompletion(false, null, null, toolChangeZMm, -1.0f);
                    res = AskUserForInput("Установите фрезу диаметром " + cuttingToolDiameterMm.ToString("0.0") + " мм.", new List<string> { "Готово", "Отмена" }, 2, false, false, false);
                    if (res > 0)
                    {
                        throw new Exception("Операция отменена.");
                    }

                    // Запускаем шпиндель
                    SetSpindleRpm(cuttingRpm, true);

                    // Цикл
                    int totalPoints = contoursToCutMm.Sum((e) => e.Length);
                    int processedPoints = 0;
                    for (int lc = 0; lc < layerCount; lc++)
                    {
                        workZMm = safeZMm - (lc * layerHeightMm);
                        i = 0;
                        for (i = 0; i < contoursToCutMm.Count; i++)
                        {
                            for (int j = 0; j < contoursToCutMm[i].Length; j++)
                            {
                                // Компенсировать криво лежащую на рабочем поле панель
                                Vector2 tgtP = new Vector2(contoursToCutMm[i][j].X, contoursToCutMm[i][j].Y);

                                // Переместиться в эту точку
                                MoveToAbsolutePositionAndWaitForCompletion((j > 0), tgtP.X, tgtP.Y, null, (j > 0) ? cuttingRateMmPerMin : -1.0f);

                                // Первая?
                                if (j == 0)
                                {
                                    // Опустить шпиндель
                                    MoveToAbsolutePositionAndWaitForCompletion(true, null, null, workZMm, zSinkRateBigMmPerMin);
                                }

                                // Прогресс и отмена
                                processedPoints++;
                                drillingProgressPercent = 100;
                                int cuttingThisLayerProgressPercent = (int)(((float)processedPoints * 100f) / (float)totalPoints);
                                cuttingProgressPercent = (int)(((lc * 100.0f) / ((float)layerCount)) + (((float)cuttingThisLayerProgressPercent) / ((float)layerCount)));
                                transformProgressPercent = (int)Math.Clamp((drillingProgressPercent * 0.5f) + (cuttingProgressPercent * 0.5f), 0f, 100f);
                                StateChanged.Invoke(this, new EventArgs());
                                if (shouldCancelOp)
                                {
                                    break;
                                }
                            }

                            // Контур завершен, поднять шпиндель
                            MoveToAbsolutePositionAndWaitForCompletion(true, null, null, safeZMm, zRetractRateBigMmPerMin);

                            // Отмена
                            if (shouldCancelOp)
                            {
                                break;
                            }
                        }
                    }

                    // Останавливаем шпиндель
                    SetSpindleRpm(0, true);
                }

                // Поднимаемся на высоту смены инструмента и центр рабочего поля
                MoveToAbsolutePositionAndWaitForCompletion(false, null, null, probeZMm, -1.0f);
                MoveToAbsolutePositionAndWaitForCompletion(false, maxTravelMmX / 2.0f, maxTravelMmY / 2.0f, toolChangeZMm, -1.0f);

                // Finale
                transformPending = false;
                SetBusyWithLongTask(false);
                StateChanged.Invoke(this, new EventArgs());
            }
            catch (Exception exc)
            {
                transformPending = false;
                SetBusyWithLongTask(false);
                ErrorOccurred.Invoke(this, new ErrorEventArgs(exc.Message));
            }
        }

        public void StartTransformProcess()
        {
            if (!IsConnected() || IsFault())
            {
                throw new Exception("Нет связи со станком или он в состоянии Alarm.");
            }
            if (IsBusyWithLongTask() || spindleActive)
            {
                throw new Exception("Станок занят другим процессом или активен шпиндель, действие недоступно.");
            }
            if (contoursToCutMm.Count <= 0 && holesToDrill.Count <= 0)
            {
                throw new Exception("Преобразования не рассчитаны.");
            }
            shouldCancelOp = false;
            workThread = new Thread(new ThreadStart(delegate
            {
                DoTransformProcess();
            }));
            workThread?.Start();
        }

        public bool IsTransformPending()
        {
            return (workThread?.IsAlive == true && transformPending);
        }
    }
}
