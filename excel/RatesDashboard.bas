Attribute VB_Name = "RatesDashboard"
Option Explicit

Private Const DATA_SHEET As String = "Data"
Private Const HEDGE_SHEET As String = "Hedge"
Private Const CURVE_SHEET As String = "Curve"

' 1) Reload ..\data\processed\yields.csv into the Data sheet
Public Sub RefreshData()
    Dim csvPath As String, wb As Workbook, dst As Worksheet
    On Error GoTo Fail
    csvPath = ThisWorkbook.Path & Application.PathSeparator & ".." & Application.PathSeparator & _
              "data" & Application.PathSeparator & "processed" & Application.PathSeparator & "yields.csv"
    If Dir(csvPath) = "" Then
        MsgBox "File not found:" & vbCrLf & csvPath, vbExclamation
        Exit Sub
    End If
    Application.ScreenUpdating = False
    Set dst = ThisWorkbook.Worksheets(DATA_SHEET)
    Set wb = Workbooks.Open(csvPath, ReadOnly:=True)
    dst.Cells.ClearContents
    wb.Worksheets(1).UsedRange.Copy Destination:=dst.Range("A1")
    wb.Close SaveChanges:=False
    dst.Columns(1).NumberFormat = "yyyy-mm-dd"
    Application.ScreenUpdating = True
    MsgBox "Loaded " & (dst.Cells(dst.Rows.Count, 1).End(xlUp).Row - 1) & " rows.", vbInformation
    Exit Sub
Fail:
    Application.ScreenUpdating = True
    MsgBox "RefreshData failed: " & Err.Description, vbCritical
End Sub

' 2) Copy the latest date, 2Y and 10Y yields from Data into the Hedge inputs
Public Sub FillFromLatest()
    Dim ds As Worksheet, hs As Worksheet, lastRow As Long, c2 As Variant, c10 As Variant
    Set ds = ThisWorkbook.Worksheets(DATA_SHEET)
    Set hs = ThisWorkbook.Worksheets(HEDGE_SHEET)
    lastRow = ds.Cells(ds.Rows.Count, 1).End(xlUp).Row
    c2 = Application.Match("2Y", ds.Rows(1), 0)
    c10 = Application.Match("10Y", ds.Rows(1), 0)
    If IsError(c2) Or IsError(c10) Then
        MsgBox "Could not find 2Y / 10Y headers on the Data sheet. Run RefreshData first.", vbExclamation
        Exit Sub
    End If
    hs.Range("B2").Value = ds.Cells(lastRow, 1).Value
    hs.Range("B3").Value = ds.Cells(lastRow, c2).Value
    hs.Range("B4").Value = ds.Cells(lastRow, c10).Value
    ComputeHedge
End Sub

' 3) DV01-neutral 2s10s steepener hedge ratio from the input cells
Public Sub ComputeHedge()
    Dim ws As Worksheet, wf As WorksheetFunction
    Dim settle As Date, mat2 As Date, mat10 As Date
    Dim y2 As Double, y10 As Double, n2 As Double
    Dim md2 As Double, md10 As Double, p2 As Double, p10 As Double
    Dim dv2 As Double, dv10 As Double, n10 As Double
    On Error GoTo Fail
    Set ws = ThisWorkbook.Worksheets(HEDGE_SHEET)
    Set wf = Application.WorksheetFunction
    settle = ws.Range("B2").Value
    y2 = ws.Range("B3").Value / 100#
    y10 = ws.Range("B4").Value / 100#
    n2 = ws.Range("B5").Value
    mat2 = DateAdd("yyyy", 2, settle)
    mat10 = DateAdd("yyyy", 10, settle)

    ' par bonds: coupon = yield. Basis 0 = US 30/360, frequency 2 = semiannual
    md2 = wf.MDuration(settle, mat2, y2, y2, 2, 0)
    md10 = wf.MDuration(settle, mat10, y10, y10, 2, 0)
    p2 = wf.Price(settle, mat2, y2, y2, 100, 2, 0)
    p10 = wf.Price(settle, mat10, y10, y10, 100, 2, 0)
    dv2 = md2 * p2 * 0.0001           ' price change per 100 face per 1bp
    dv10 = md10 * p10 * 0.0001
    n10 = n2 * dv2 / dv10

    ws.Range("B8").Value = md2
    ws.Range("B9").Value = md10
    ws.Range("B10").Value = dv2
    ws.Range("B11").Value = dv10
    ws.Range("B12").Value = n10
    ws.Range("B13").Value = n10 / n2
    ws.Range("B14").Value = n2 / 100# * dv2 - n10 / 100# * dv10
    ws.Range("B12").NumberFormat = "#,##0"
    ws.Range("B8:B11").NumberFormat = "0.0000"
    ws.Range("B13").NumberFormat = "0.0000"
    ws.Range("B14").NumberFormat = "0.00"
    Exit Sub
Fail:
    MsgBox "ComputeHedge failed: " & Err.Description, vbCritical
End Sub

' 4) Chart the latest curve against 1 month ago and 1 year ago
Public Sub ChartCurve()
    Dim ds As Worksheet, cs As Worksheet, lastRow As Long, lastCol As Long
    Dim rMonth As Long, rYear As Long, c As Long, co As ChartObject
    Set ds = ThisWorkbook.Worksheets(DATA_SHEET)
    Set cs = ThisWorkbook.Worksheets(CURVE_SHEET)
    lastRow = ds.Cells(ds.Rows.Count, 1).End(xlUp).Row
    lastCol = ds.Cells(1, ds.Columns.Count).End(xlToLeft).Column
    If lastRow < 3 Then
        MsgBox "No data. Run RefreshData first.", vbExclamation
        Exit Sub
    End If
    rMonth = RowOnOrBefore(ds, lastRow, ds.Cells(lastRow, 1).Value - 30)
    rYear = RowOnOrBefore(ds, lastRow, ds.Cells(lastRow, 1).Value - 365)

    For Each co In cs.ChartObjects
        co.Delete
    Next co
    cs.Cells.Clear
    cs.Range("A1").Value = "Tenor"
    cs.Range("B1").Value = "Latest " & Format(ds.Cells(lastRow, 1).Value, "yyyy-mm-dd")
    cs.Range("C1").Value = "1M ago " & Format(ds.Cells(rMonth, 1).Value, "yyyy-mm-dd")
    cs.Range("D1").Value = "1Y ago " & Format(ds.Cells(rYear, 1).Value, "yyyy-mm-dd")
    For c = 2 To lastCol
        cs.Cells(c, 1).Value = ds.Cells(1, c).Value
        cs.Cells(c, 2).Value = ds.Cells(lastRow, c).Value
        cs.Cells(c, 3).Value = ds.Cells(rMonth, c).Value
        cs.Cells(c, 4).Value = ds.Cells(rYear, c).Value
    Next c

    Set co = cs.ChartObjects.Add(Left:=300, Top:=10, Width:=540, Height:=330)
    With co.Chart
        .ChartType = xlLineMarkers
        .SetSourceData Source:=cs.Range(cs.Cells(1, 1), cs.Cells(lastCol, 4)), PlotBy:=xlColumns
        .HasTitle = True
        .ChartTitle.Text = "US Treasury par curve: now vs 1 month vs 1 year ago"
        .Axes(xlValue).HasTitle = True
        .Axes(xlValue).AxisTitle.Text = "Yield (%)"
    End With
End Sub

' Helper: last row whose date is <= target (Data is sorted oldest to newest)
Private Function RowOnOrBefore(ws As Worksheet, lastRow As Long, target As Double) As Long
    Dim r As Long
    For r = lastRow To 2 Step -1
        If ws.Cells(r, 1).Value <= target Then
            RowOnOrBefore = r
            Exit Function
        End If
    Next r
    RowOnOrBefore = 2
End Function
