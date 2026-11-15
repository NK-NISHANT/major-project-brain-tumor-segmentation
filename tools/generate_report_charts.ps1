$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$results = Join-Path $root "results"
$figureDir = Join-Path $root "tmp\report_figures"
New-Item -ItemType Directory -Path $figureDir -Force | Out-Null

$models = @(
    @{ Key = "baseline_unet2d_5ep"; Name = "U-Net"; DataName = "UNet"; Color = 0xD07A28 },
    @{ Key = "unetpp2d_5ep"; Name = "U-Net++"; DataName = "UNetPlusPlus"; Color = 0x5DAB36 },
    @{ Key = "attention_unet2d_5ep"; Name = "Attention U-Net"; DataName = "AttentionUNet"; Color = 0x5B5BD3 }
)

$history = @{}
foreach ($model in $models) {
    $rows = Import-Csv (Join-Path $results "$($model.Key)\history.csv")
    $history[$model.Name] = $rows
    $model.Best = $rows | Sort-Object { [double]$_.mean_dice } -Descending | Select-Object -First 1
}

$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$workbook = $excel.Workbooks.Add()

function Set-Cell($sheet, [int]$row, [int]$column, $value) {
    $sheet.Cells.Item($row, $column).Formula = [string]$value
}

function Set-ChartStyle($chart, [string]$title, [string]$axisTitle, [double]$axisMax, [double]$majorUnit) {
    $chart.HasTitle = $true
    $chart.ChartTitle.Text = $title
    $chart.ChartTitle.Format.TextFrame2.TextRange.Font.Name = "Arial"
    $chart.ChartTitle.Format.TextFrame2.TextRange.Font.Size = 16
    $chart.ChartTitle.Format.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = 0x503826
    $chart.HasLegend = $true
    $chart.Legend.Position = -4107
    $chart.ChartArea.Format.Fill.ForeColor.RGB = 0xFFFFFF
    $chart.PlotArea.Format.Fill.ForeColor.RGB = 0xFFFFFF
    $axis = $chart.Axes(2)
    $axis.HasTitle = $true
    $axis.AxisTitle.Text = $axisTitle
    $axis.MinimumScale = 0
    $axis.MaximumScale = $axisMax
    $axis.MajorUnit = $majorUnit
    $axis.TickLabels.Font.Name = "Arial"
    $axis.TickLabels.Font.Size = 10
    $axis.TickLabels.Font.Color = 0x503826
    $axis.MajorGridlines.Format.Line.ForeColor.RGB = 0xEBE5DF
    $categoryAxis = $chart.Axes(1)
    $categoryAxis.TickLabels.Font.Name = "Arial"
    $categoryAxis.TickLabels.Font.Size = 10
    $categoryAxis.TickLabels.Font.Color = 0x503826
}

function Export-Chart($sheet, [string]$rangeAddress, [int]$chartType, [string]$title, [string]$axisTitle, [double]$axisMax, [double]$majorUnit, [string]$filename) {
    $chartObject = $sheet.ChartObjects().Add(18, 150, 900, 520)
    $chart = $chartObject.Chart
    $chart.ChartType = $chartType
    $null = $chart.SetSourceData($sheet.Range($rangeAddress))
    $chart.PlotBy = 2
    $null = Set-ChartStyle $chart $title $axisTitle $axisMax $majorUnit
    return ,$chart
}

function Save-Chart($chart, [string]$filename) {
    $null = $chart.Export((Join-Path $figureDir $filename), "PNG")
}

try {
    while ($workbook.Worksheets.Count -lt 5) {
        $null = $workbook.Worksheets.Add()
    }
    $sheets = @()
    for ($i = 1; $i -le 5; $i++) {
        $sheet = $workbook.Worksheets.Item($i)
        $sheet.Name = "Chart$i"
        $sheets += $sheet
    }

    $metricSheet = $sheets[0]
    Set-Cell $metricSheet 1 1 "Model"
    Set-Cell $metricSheet 1 2 "Best Dice"
    Set-Cell $metricSheet 1 3 "Best IoU"
    for ($i = 0; $i -lt $models.Count; $i++) {
        $row = $i + 2
        $model = $models[$i]
        Set-Cell $metricSheet $row 1 $model.Name
        Set-Cell $metricSheet $row 2 ([double]$model.Best.mean_dice)
        Set-Cell $metricSheet $row 3 ([double]$model.Best.mean_iou)
    }
    $metricChart = Export-Chart $metricSheet "A1:C4" 51 "Best Validation Scores by Model" "Score" 1 0.2 "model_metrics.png"
    foreach ($seriesIndex in 1..2) {
        $series = $metricChart.SeriesCollection($seriesIndex)
        $series.Format.Fill.ForeColor.RGB = if ($seriesIndex -eq 1) { 0xD07A28 } else { 0x5DAB36 }
        $series.Format.Line.ForeColor.RGB = 0x503826
        $series.ApplyDataLabels(2)
        $series.DataLabels().ShowValue = $true
        $series.DataLabels().NumberFormat = "0.000"
        $series.DataLabels().Font.Name = "Arial"
        $series.DataLabels().Font.Size = 10
    }
    Save-Chart $metricChart "model_metrics.png"

    $lossSheet = $sheets[1]
    $headers = @("Epoch", "U-Net Train", "U-Net Validation", "U-Net++ Train", "U-Net++ Validation", "Attention U-Net Train", "Attention U-Net Validation")
    for ($column = 1; $column -le $headers.Count; $column++) { Set-Cell $lossSheet 1 $column $headers[$column - 1] }
    for ($epoch = 1; $epoch -le 5; $epoch++) {
        Set-Cell $lossSheet ($epoch + 1) 1 $epoch
        for ($modelIndex = 0; $modelIndex -lt $models.Count; $modelIndex++) {
            $rowData = $history[$models[$modelIndex].Name] | Where-Object { [int][double]$_.epoch -eq $epoch } | Select-Object -First 1
            Set-Cell $lossSheet ($epoch + 1) (2 + 2 * $modelIndex) ([double]$rowData.training_loss)
            Set-Cell $lossSheet ($epoch + 1) (3 + 2 * $modelIndex) ([double]$rowData.validation_loss)
        }
    }
    $lossChart = Export-Chart $lossSheet "A1:G6" 65 "Training and Validation Loss by Epoch" "Loss" 1.6 0.2 "loss_curves.png"
    $lossChart.SeriesCollection(1).Delete()
    $lossChart.Axes(1).HasTitle = $true
    $lossChart.Axes(1).AxisTitle.Text = "Epoch"
    for ($i = 1; $i -le 6; $i++) {
        $series = $lossChart.SeriesCollection($i)
        $model = $models[[math]::Floor(($i - 1) / 2)]
        $series.Format.Line.ForeColor.RGB = $model.Color
        $series.Format.Line.Weight = 2
        $series.MarkerStyle = 8
        $series.MarkerSize = 6
        if ($i % 2 -eq 0) { $series.Format.Line.DashStyle = 4 }
    }
    $lossChart.Legend.Font.Name = "Arial"
    $lossChart.Legend.Font.Size = 9
    $lossChart.Legend.Font.Color = 0x503826
    Save-Chart $lossChart "loss_curves.png"

    $diceSheet = $sheets[2]
    Set-Cell $diceSheet 1 1 "Epoch"
    for ($i = 0; $i -lt $models.Count; $i++) { Set-Cell $diceSheet 1 ($i + 2) $models[$i].Name }
    for ($epoch = 1; $epoch -le 5; $epoch++) {
        Set-Cell $diceSheet ($epoch + 1) 1 $epoch
        for ($modelIndex = 0; $modelIndex -lt $models.Count; $modelIndex++) {
            $rowData = $history[$models[$modelIndex].Name] | Where-Object { [int][double]$_.epoch -eq $epoch } | Select-Object -First 1
            Set-Cell $diceSheet ($epoch + 1) ($modelIndex + 2) ([double]$rowData.mean_dice)
        }
    }
    $diceChart = Export-Chart $diceSheet "A1:D6" 65 "Validation Mean Dice by Epoch" "Mean Dice" 1 0.2 "mean_dice_trend.png"
    $diceChart.SeriesCollection(1).Delete()
    $diceChart.Axes(1).HasTitle = $true
    $diceChart.Axes(1).AxisTitle.Text = "Epoch"
    for ($i = 1; $i -le 3; $i++) {
        $series = $diceChart.SeriesCollection($i)
        $series.Format.Line.ForeColor.RGB = $models[$i - 1].Color
        $series.Format.Line.Weight = 2.5
        $series.MarkerStyle = 8
        $series.MarkerSize = 7
    }
    $diceChart.Legend.Font.Name = "Arial"
    $diceChart.Legend.Font.Size = 10
    $diceChart.Legend.Font.Color = 0x503826
    Save-Chart $diceChart "mean_dice_trend.png"

    $classSheet = $sheets[3]
    Set-Cell $classSheet 1 1 "Tumor Label"
    for ($i = 0; $i -lt $models.Count; $i++) { Set-Cell $classSheet 1 ($i + 2) $models[$i].Name }
    for ($label = 1; $label -le 3; $label++) {
        Set-Cell $classSheet ($label + 1) 1 "Class $label"
        for ($modelIndex = 0; $modelIndex -lt $models.Count; $modelIndex++) {
            $model = $models[$modelIndex]
            $classScore = $model.Best.PSObject.Properties["dice_class_$label"].Value
            Set-Cell $classSheet ($label + 1) ($modelIndex + 2) ([double]$classScore)
        }
    }
    $classChart = Export-Chart $classSheet "A1:D4" 51 "Foreground Dice at Each Model's Best Epoch" "Dice" 1 0.2 "per_class_dice.png"
    for ($i = 1; $i -le 3; $i++) {
        $series = $classChart.SeriesCollection($i)
        $series.Format.Fill.ForeColor.RGB = $models[$i - 1].Color
        $series.Format.Line.ForeColor.RGB = 0x503826
        $series.ApplyDataLabels(2)
        $series.DataLabels().ShowValue = $true
        $series.DataLabels().NumberFormat = "0.000"
        $series.DataLabels().Font.Name = "Arial"
        $series.DataLabels().Font.Size = 9
    }
    Save-Chart $classChart "per_class_dice.png"

    $missSheet = $sheets[4]
    Set-Cell $missSheet 1 1 "Model"
    Set-Cell $missSheet 1 2 "Complete Miss Rate"
    $source = Import-Csv (Join-Path $results "uncertainty_unet\model_uncertainty_comparison.csv")
    $foregroundRows = $source | Where-Object { $_.has_foreground -eq "True" }
    for ($i = 0; $i -lt $models.Count; $i++) {
        $model = $models[$i]
        $modelRows = @($foregroundRows | Where-Object { $_.model -eq $model.DataName })
        $misses = @($modelRows | Where-Object { [int]$_.pred_foreground_pixels -eq 0 }).Count
        $row = $i + 2
        Set-Cell $missSheet $row 1 $model.Name
        Set-Cell $missSheet $row 2 ($misses / $modelRows.Count)
    }
    $missChart = Export-Chart $missSheet "A1:B4" 51 "Complete Foreground Miss Rate" "Miss Rate" 0.2 0.05 "miss_rate.png"
    $missSeries = $missChart.SeriesCollection(1)
    $missSeries.Format.Fill.ForeColor.RGB = 0xD07A28
    $missSeries.Format.Line.ForeColor.RGB = 0x503826
    $missSeries.ApplyDataLabels(2)
    $missSeries.DataLabels().ShowValue = $true
    $missSeries.DataLabels().NumberFormat = "0.00%"
    $missSeries.DataLabels().Font.Name = "Arial"
    $missSeries.DataLabels().Font.Size = 10
    $missChart.Axes(2).TickLabels.NumberFormat = "0%"
    $missChart.HasLegend = $false
    Save-Chart $missChart "miss_rate.png"

    Get-ChildItem $figureDir -Filter "*.png" | Select-Object -ExpandProperty FullName
}
finally {
    $workbook.Close($false)
    $excel.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($workbook)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
