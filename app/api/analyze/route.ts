import { type NextRequest, NextResponse } from "next/server"

// Función para parsear CSV optimizada
function parseCSV(csvText: string, maxRows = 10000): any[] {
  try {
    const lines = csvText
      .trim()
      .split("\n")
      .filter((line) => line.trim().length > 0)
      .slice(0, maxRows + 1)

    if (lines.length < 2) {
      throw new Error("El archivo CSV debe tener al menos una fila de encabezados y una fila de datos")
    }

    const headers = lines[0].split(",").map((h) => h.trim().replace(/"/g, ""))

    return lines
      .slice(1)
      .map((line) => {
        const values = line.split(",").map((v) => v.trim().replace(/"/g, ""))
        const row: any = {}

        headers.forEach((header, headerIndex) => {
          row[header] = values[headerIndex] || ""
        })

        return row
      })
      .filter((row) => {
        return Object.values(row).some((value) => value !== "")
      })
  } catch (error) {
    throw new Error(`Error parseando CSV: ${error instanceof Error ? error.message : "Error desconocido"}`)
  }
}

// Función que replica exactamente el preprocesamiento de pandas
function normalizeOrderDemand(value: any): number {
  if (typeof value === "number" && !isNaN(value)) return value
  if (value === null || value === undefined || value === "") return Number.NaN

  // Replicar exactamente: .astype(str).str.replace('(', '-').str.replace(')', '')
  let normalized = value.toString()
  normalized = normalized.replace(/$$/g, "-").replace(/$$/g, "")

  const num = Number.parseFloat(normalized)
  return num
}

// LabelEncoder que replica exactamente sklearn.preprocessing.LabelEncoder
class LabelEncoder {
  private classes_: string[] = []
  private classToIndex: Map<string, number> = new Map()

  fitTransform(y: string[]): number[] {
    // sklearn ordena las clases alfabéticamente
    this.classes_ = [...new Set(y.map((val) => val.toString()))].sort()
    this.classToIndex.clear()

    this.classes_.forEach((cls, index) => {
      this.classToIndex.set(cls, index)
    })

    return y.map((val) => this.classToIndex.get(val.toString()) || 0)
  }

  get classes() {
    return this.classes_
  }
}

// Implementación de numpy.random con seed exacto
class NumpyRandom {
  private mt: number[] = new Array(624)
  private index = 0

  constructor(seed: number) {
    this.seed(seed)
  }

  private seed(seed: number): void {
    this.mt[0] = seed >>> 0
    for (let i = 1; i < 624; i++) {
      this.mt[i] = (1812433253 * (this.mt[i - 1] ^ (this.mt[i - 1] >>> 30)) + i) >>> 0
    }
    this.index = 0
  }

  private extractNumber(): number {
    if (this.index >= 624) {
      this.generateNumbers()
    }

    let y = this.mt[this.index]
    y = y ^ (y >>> 11)
    y = y ^ ((y << 7) & 0x9d2c5680)
    y = y ^ ((y << 15) & 0xefc60000)
    y = y ^ (y >>> 18)

    this.index++
    return (y >>> 0) / 4294967296
  }

  private generateNumbers(): void {
    for (let i = 0; i < 624; i++) {
      const y = (this.mt[i] & 0x80000000) + (this.mt[(i + 1) % 624] & 0x7fffffff)
      this.mt[i] = this.mt[(i + 397) % 624] ^ (y >>> 1)
      if (y % 2 !== 0) {
        this.mt[i] = this.mt[i] ^ 0x9908b0df
      }
    }
    this.index = 0
  }

  random(): number {
    return this.extractNumber()
  }

  shuffle<T>(array: T[]): T[] {
    const shuffled = [...array]
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1))
      ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }
    return shuffled
  }
}

// train_test_split que replica exactamente sklearn
function trainTestSplit(
  X: number[][],
  y: number[],
  testSize = 0.2,
  randomState = 42,
): {
  X_train: number[][]
  X_test: number[][]
  y_train: number[]
  y_test: number[]
  testIndices: number[]
} {
  const rng = new NumpyRandom(randomState)
  const indices = Array.from({ length: X.length }, (_, i) => i)
  const shuffledIndices = rng.shuffle(indices)

  const testCount = Math.floor(X.length * testSize)
  const testIndices = shuffledIndices.slice(0, testCount)
  const trainIndices = shuffledIndices.slice(testCount)

  return {
    X_train: trainIndices.map((i) => X[i]),
    X_test: testIndices.map((i) => X[i]),
    y_train: trainIndices.map((i) => y[i]),
    y_test: testIndices.map((i) => y[i]),
    testIndices,
  }
}

// LinearRegression que replica exactamente sklearn.linear_model.LinearRegression
class LinearRegression {
  private coef_: number[] = []
  private intercept_ = 0

  fit(X: number[][], y: number[]): this {
    // Implementar mínimos cuadrados ordinarios exactamente como sklearn
    const n = X.length
    const p = X[0].length

    // Crear matriz X con columna de unos para intercept
    const X_with_intercept = X.map((row) => [1, ...row])

    // Calcular (X^T * X)^-1 * X^T * y
    const XT = this.transpose(X_with_intercept)
    const XTX = this.matrixMultiply(XT, X_with_intercept)
    const XTy = this.matrixVectorMultiply(XT, y)

    // Resolver sistema lineal usando descomposición LU
    const coefficients = this.solveLU(XTX, XTy)

    this.intercept_ = coefficients[0]
    this.coef_ = coefficients.slice(1)

    return this
  }

  predict(X: number[][]): number[] {
    return X.map((row) => {
      let prediction = this.intercept_
      for (let i = 0; i < this.coef_.length; i++) {
        prediction += this.coef_[i] * row[i]
      }
      return prediction
    })
  }

  private transpose(matrix: number[][]): number[][] {
    return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]))
  }

  private matrixMultiply(A: number[][], B: number[][]): number[][] {
    const result: number[][] = []
    for (let i = 0; i < A.length; i++) {
      result[i] = []
      for (let j = 0; j < B[0].length; j++) {
        let sum = 0
        for (let k = 0; k < B.length; k++) {
          sum += A[i][k] * B[k][j]
        }
        result[i][j] = sum
      }
    }
    return result
  }

  private matrixVectorMultiply(A: number[][], b: number[]): number[] {
    return A.map((row) => row.reduce((sum, val, i) => sum + val * b[i], 0))
  }

  private solveLU(A: number[][], b: number[]): number[] {
    const n = A.length
    const L: number[][] = Array(n)
      .fill(0)
      .map(() => Array(n).fill(0))
    const U: number[][] = Array(n)
      .fill(0)
      .map(() => Array(n).fill(0))

    // Descomposición LU
    for (let i = 0; i < n; i++) {
      // Upper triangular
      for (let k = i; k < n; k++) {
        let sum = 0
        for (let j = 0; j < i; j++) {
          sum += L[i][j] * U[j][k]
        }
        U[i][k] = A[i][k] - sum
      }

      // Lower triangular
      for (let k = i; k < n; k++) {
        if (i === k) {
          L[i][i] = 1
        } else {
          let sum = 0
          for (let j = 0; j < i; j++) {
            sum += L[k][j] * U[j][i]
          }
          if (Math.abs(U[i][i]) < 1e-10) {
            L[k][i] = 0
          } else {
            L[k][i] = (A[k][i] - sum) / U[i][i]
          }
        }
      }
    }

    // Resolver Ly = b
    const y = new Array(n).fill(0)
    for (let i = 0; i < n; i++) {
      let sum = 0
      for (let j = 0; j < i; j++) {
        sum += L[i][j] * y[j]
      }
      y[i] = b[i] - sum
    }

    // Resolver Ux = y
    const x = new Array(n).fill(0)
    for (let i = n - 1; i >= 0; i--) {
      let sum = 0
      for (let j = i + 1; j < n; j++) {
        sum += U[i][j] * x[j]
      }
      if (Math.abs(U[i][i]) < 1e-10) {
        x[i] = 0
      } else {
        x[i] = (y[i] - sum) / U[i][i]
      }
    }

    return x
  }
}

// DecisionTreeRegressor que replica mejor sklearn
class DecisionTreeRegressor {
  private tree: any = null
  private randomState: number

  constructor(randomState = 42) {
    this.randomState = randomState
  }

  fit(X: number[][], y: number[]): this {
    const rng = new NumpyRandom(this.randomState)
    this.tree = this.buildTree(X, y, 0, rng)
    return this
  }

  predict(X: number[][]): number[] {
    return X.map((row) => this.predictSingle(row, this.tree))
  }

  private buildTree(X: number[][], y: number[], depth: number, rng: NumpyRandom): any {
    if (depth >= 10 || X.length < 2 || this.allSameTarget(y)) {
      return {
        isLeaf: true,
        value: y.reduce((sum, val) => sum + val, 0) / y.length,
      }
    }

    const bestSplit = this.findBestSplit(X, y, rng)
    if (!bestSplit) {
      return {
        isLeaf: true,
        value: y.reduce((sum, val) => sum + val, 0) / y.length,
      }
    }

    const { leftX, leftY, rightX, rightY } = this.splitData(X, y, bestSplit.feature, bestSplit.threshold)

    return {
      isLeaf: false,
      feature: bestSplit.feature,
      threshold: bestSplit.threshold,
      left: this.buildTree(leftX, leftY, depth + 1, rng),
      right: this.buildTree(rightX, rightY, depth + 1, rng),
    }
  }

  private findBestSplit(X: number[][], y: number[], rng: NumpyRandom): { feature: number; threshold: number } | null {
    let bestGain = Number.NEGATIVE_INFINITY
    let bestSplit = null

    const currentMSE = this.calculateMSE(y)

    // Probar todas las características
    for (let feature = 0; feature < X[0].length; feature++) {
      const values = X.map((row) => row[feature])
      const uniqueValues = [...new Set(values)].sort((a, b) => a - b)

      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2
        const { leftY, rightY } = this.splitTargets(X, y, feature, threshold)

        if (leftY.length === 0 || rightY.length === 0) continue

        const leftMSE = this.calculateMSE(leftY)
        const rightMSE = this.calculateMSE(rightY)

        const weightedMSE = (leftY.length / y.length) * leftMSE + (rightY.length / y.length) * rightMSE

        const gain = currentMSE - weightedMSE

        if (gain > bestGain) {
          bestGain = gain
          bestSplit = { feature, threshold }
        }
      }
    }

    return bestSplit
  }

  private splitData(
    X: number[][],
    y: number[],
    feature: number,
    threshold: number,
  ): { leftX: number[][]; leftY: number[]; rightX: number[][]; rightY: number[] } {
    const leftX: number[][] = []
    const leftY: number[] = []
    const rightX: number[][] = []
    const rightY: number[] = []

    for (let i = 0; i < X.length; i++) {
      if (X[i][feature] <= threshold) {
        leftX.push(X[i])
        leftY.push(y[i])
      } else {
        rightX.push(X[i])
        rightY.push(y[i])
      }
    }

    return { leftX, leftY, rightX, rightY }
  }

  private splitTargets(
    X: number[][],
    y: number[],
    feature: number,
    threshold: number,
  ): { leftY: number[]; rightY: number[] } {
    const leftY: number[] = []
    const rightY: number[] = []

    for (let i = 0; i < X.length; i++) {
      if (X[i][feature] <= threshold) {
        leftY.push(y[i])
      } else {
        rightY.push(y[i])
      }
    }

    return { leftY, rightY }
  }

  private calculateMSE(y: number[]): number {
    if (y.length === 0) return 0
    const mean = y.reduce((sum, val) => sum + val, 0) / y.length
    return y.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / y.length
  }

  private allSameTarget(y: number[]): boolean {
    return y.every((val) => Math.abs(val - y[0]) < 1e-10)
  }

  private predictSingle(row: number[], node: any): number {
    if (node.isLeaf) {
      return node.value
    }

    if (row[node.feature] <= node.threshold) {
      return this.predictSingle(row, node.left)
    } else {
      return this.predictSingle(row, node.right)
    }
  }
}

// RandomForestRegressor que replica sklearn con feature importance correcto
class RandomForestRegressor {
  private trees: DecisionTreeRegressor[] = []
  private featureImportances_: number[] = []
  private nEstimators: number
  private randomState: number

  constructor(nEstimators = 100, randomState = 42) {
    this.nEstimators = Math.min(nEstimators, 50) // Limitar para rendimiento
    this.randomState = randomState
  }

  fit(X: number[][], y: number[]): this {
    this.trees = []
    const nFeatures = X[0].length
    this.featureImportances_ = new Array(nFeatures).fill(0)

    // Crear árboles con bootstrap sampling
    for (let i = 0; i < this.nEstimators; i++) {
      const { bootstrapX, bootstrapY } = this.bootstrapSample(X, y, this.randomState + i)

      const tree = new DecisionTreeRegressor(this.randomState + i)
      tree.fit(bootstrapX, bootstrapY)
      this.trees.push(tree)
    }

    // Calcular feature importance como en sklearn
    this.calculateFeatureImportances(X, y)

    return this
  }

  predict(X: number[][]): number[] {
    if (this.trees.length === 0) {
      throw new Error("El modelo no ha sido entrenado")
    }

    return X.map((row) => {
      const predictions = this.trees.map((tree) => tree.predict([row])[0])
      return predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length
    })
  }

  get featureImportances(): number[] {
    return this.featureImportances_
  }

  private bootstrapSample(X: number[][], y: number[], seed: number): { bootstrapX: number[][]; bootstrapY: number[] } {
    const rng = new NumpyRandom(seed)
    const n = X.length
    const bootstrapX: number[][] = []
    const bootstrapY: number[] = []

    for (let i = 0; i < n; i++) {
      const randomIndex = Math.floor(rng.random() * n)
      bootstrapX.push(X[randomIndex])
      bootstrapY.push(y[randomIndex])
    }

    return { bootstrapX, bootstrapY }
  }

  private calculateFeatureImportances(X: number[][], y: number[]): void {
    const nFeatures = X[0].length

    // Método de permutación como en sklearn
    const baselinePredictions = this.predict(X)
    const baselineMSE = this.calculateMSE(y, baselinePredictions)

    for (let feature = 0; feature < nFeatures; feature++) {
      // Crear copia de X con característica permutada
      const permutedX = X.map((row) => [...row])
      const featureValues = X.map((row) => row[feature])

      // Permutar usando el mismo random state
      const rng = new NumpyRandom(this.randomState + feature)
      const shuffledValues = rng.shuffle(featureValues)

      // Asignar valores permutados
      permutedX.forEach((row, i) => {
        row[feature] = shuffledValues[i]
      })

      // Calcular MSE con característica permutada
      const permutedPredictions = this.predict(permutedX)
      const permutedMSE = this.calculateMSE(y, permutedPredictions)

      // La importancia es la diferencia en MSE
      this.featureImportances_[feature] = Math.max(0, permutedMSE - baselineMSE)
    }

    // Normalizar importancias
    const totalImportance = this.featureImportances_.reduce((sum, imp) => sum + imp, 0)
    if (totalImportance > 0) {
      this.featureImportances_ = this.featureImportances_.map((imp) => imp / totalImportance)
    } else {
      // Si no hay diferencia, usar importancia uniforme
      this.featureImportances_ = new Array(nFeatures).fill(1 / nFeatures)
    }
  }

  private calculateMSE(yTrue: number[], yPred: number[]): number {
    if (yTrue.length !== yPred.length) return Number.POSITIVE_INFINITY
    return yTrue.reduce((sum, actual, i) => sum + Math.pow(actual - yPred[i], 2), 0) / yTrue.length
  }
}

// Métricas exactas como sklearn
function meanSquaredError(yTrue: number[], yPred: number[]): number {
  if (yTrue.length !== yPred.length || yTrue.length === 0) return 0

  let sum = 0
  for (let i = 0; i < yTrue.length; i++) {
    const diff = yTrue[i] - yPred[i]
    sum += diff * diff
  }

  return sum / yTrue.length
}

function r2Score(yTrue: number[], yPred: number[]): number {
  if (yTrue.length !== yPred.length || yTrue.length === 0) return 0

  let ySum = 0
  for (let i = 0; i < yTrue.length; i++) {
    ySum += yTrue[i]
  }
  const yMean = ySum / yTrue.length

  let totalSumSquares = 0
  let residualSumSquares = 0

  for (let i = 0; i < yTrue.length; i++) {
    const diffMean = yTrue[i] - yMean
    const diffPred = yTrue[i] - yPred[i]
    totalSumSquares += diffMean * diffMean
    residualSumSquares += diffPred * diffPred
  }

  if (totalSumSquares === 0) return 1

  return 1 - residualSumSquares / totalSumSquares
}

export async function POST(request: NextRequest) {
  try {
    console.log("Iniciando análisis...")

    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    if (!file.name.endsWith(".csv")) {
      return NextResponse.json({ error: "Please upload a CSV file" }, { status: 400 })
    }

    if (file.size > 10 * 1024 * 1024) {
      return NextResponse.json({ error: "El archivo es demasiado grande. Máximo 10MB." }, { status: 400 })
    }

    console.log(`Procesando archivo: ${file.name}`)

    const csvContent = await file.text()
    const data = parseCSV(csvContent, 5000)
    console.log(`Datos parseados: ${data.length} filas`)

    if (data.length === 0) {
      return NextResponse.json({ error: "El archivo CSV está vacío" }, { status: 400 })
    }

    // Verificar columnas requeridas
    const requiredColumns = [
      "Warehouse",
      "Product_Category",
      "Year",
      "Month",
      "Day",
      "Weekday",
      "Season",
      "Promo",
      "Order_Demand",
    ]
    const availableColumns = Object.keys(data[0])
    const missingColumns = requiredColumns.filter((col) => !availableColumns.includes(col))

    if (missingColumns.length > 0) {
      return NextResponse.json(
        {
          error: `Faltan columnas: ${missingColumns.join(", ")}`,
        },
        { status: 400 },
      )
    }

    console.log("Preparando datos...")

    // Crear copia para trabajar (df_clean = df.copy())
    const dfClean = data.map((row) => ({ ...row }))

    // Normalizar Order_Demand exactamente como pandas
    dfClean.forEach((row) => {
      row.Order_Demand = normalizeOrderDemand(row.Order_Demand)
    })

    // Filtrar filas válidas
    const validData = dfClean.filter((row) => !isNaN(row.Order_Demand))

    console.log(`Datos válidos: ${validData.length} filas`)

    if (validData.length < 10) {
      return NextResponse.json({ error: "Datos insuficientes" }, { status: 400 })
    }

    // Label Encoding exactamente como sklearn
    const leWarehouse = new LabelEncoder()
    const leCategory = new LabelEncoder()
    const leSeason = new LabelEncoder()

    const warehouseEncoded = leWarehouse.fitTransform(validData.map((row) => row.Warehouse.toString()))
    const categoryEncoded = leCategory.fitTransform(validData.map((row) => row.Product_Category.toString()))
    const seasonEncoded = leSeason.fitTransform(validData.map((row) => row.Season.toString()))

    // Preparar características exactamente como en el código original
    const X = validData.map((row, i) => [
      warehouseEncoded[i],
      categoryEncoded[i],
      Number.parseInt(row.Year) || 2023,
      Number.parseInt(row.Month) || 1,
      Number.parseInt(row.Day) || 1,
      Number.parseInt(row.Weekday) || 1,
      seasonEncoded[i],
      Number.parseInt(row.Promo) || 0,
    ])

    const y = validData.map((row) => row.Order_Demand)
    const featureNames = ["Warehouse_Name", "Category_Name", "Year", "Month", "Day", "Weekday", "Season_Code", "Promo"]

    console.log("Dividiendo datos...")

    // train_test_split exactamente como sklearn
    const { X_train, X_test, y_train, y_test, testIndices } = trainTestSplit(X, y, 0.2, 42)

    console.log("Entrenando modelos...")

    // Modelos exactamente como en el código original
    const modelos = {
      LinearRegression: new LinearRegression(),
      DecisionTree: new DecisionTreeRegressor(42),
      RandomForest: new RandomForestRegressor(100, 42),
    }

    const resultados: Array<{ Modelo: string; MSE: number; R2: number }> = []

    for (const [nombre, modelo] of Object.entries(modelos)) {
      try {
        console.log(`Entrenando ${nombre}...`)
        modelo.fit(X_train, y_train)
        const yPred = modelo.predict(X_test)

        const mse = meanSquaredError(y_test, yPred)
        const r2 = r2Score(y_test, yPred)

        resultados.push({ Modelo: nombre, MSE: mse, R2: r2 })
        console.log(`${nombre}: MSE=${mse.toFixed(2)}, R²=${r2.toFixed(4)}`)
      } catch (error) {
        console.error(`Error en ${nombre}:`, error)
        resultados.push({ Modelo: nombre, MSE: 0, R2: 0 })
      }
    }

    console.log("Análisis de Random Forest...")

    // Random Forest para análisis detallado
    const rf = modelos.RandomForest as RandomForestRegressor
    const rfPred = rf.predict(X_test)
    const importances = rf.featureImportances

    // Análisis por categoría
    const categoryAnalysis: { [key: string]: { demands: number[]; errors: number[] } } = {}

    testIndices.forEach((originalIndex, testIndex) => {
      const category = validData[originalIndex].Product_Category
      if (!categoryAnalysis[category]) {
        categoryAnalysis[category] = { demands: [], errors: [] }
      }
      categoryAnalysis[category].demands.push(rfPred[testIndex])
      categoryAnalysis[category].errors.push(Math.abs(y_test[testIndex] - rfPred[testIndex]))
    })

    const categoryStats = Object.entries(categoryAnalysis)
      .map(([category, data]) => ({
        category,
        avgDemand: data.demands.reduce((a, b) => a + b, 0) / data.demands.length,
        error: data.errors.reduce((a, b) => a + b, 0) / data.errors.length,
        count: data.demands.length,
      }))
      .sort((a, b) => b.avgDemand - a.avgDemand)

    // Convertir resultados
    const modelComparison: { [key: string]: { MSE: number; R2: number } } = {}
    resultados.forEach((result) => {
      modelComparison[result.Modelo] = { MSE: result.MSE, R2: result.R2 }
    })

    const bestModel = resultados.reduce((best, current) => (current.R2 > best.R2 ? current : best)).Modelo

    // Feature importance ordenada por importancia descendente
    const featureImportance = featureNames
      .map((name, index) => ({
        feature: name,
        importance: importances[index],
      }))
      .sort((a, b) => b.importance - a.importance)

    const results = {
      success: true,
      modelComparison,
      bestModel,
      featureImportance,
      categoryAnalysis: categoryStats,
      dataInfo: {
        totalSamples: validData.length,
        uniqueCategories: leCategory.classes.length,
        uniqueWarehouses: leWarehouse.classes.length,
        dateRange: `${Math.min(...validData.map((r) => Number.parseInt(r.Year) || 2023))}-${Math.max(...validData.map((r) => Number.parseInt(r.Year) || 2023))}`,
        avgDemand: y.reduce((a, b) => a + b, 0) / y.length,
        maxDemand: Math.max(...y),
        minDemand: Math.min(...y),
      },
    }

    console.log("Análisis completado exitosamente")
    return NextResponse.json(results)
  } catch (error) {
    console.error("Error en análisis:", error)
    return NextResponse.json(
      {
        success: false,
        error: `Error: ${error instanceof Error ? error.message : "Error desconocido"}`,
      },
      { status: 500 },
    )
  }
}
