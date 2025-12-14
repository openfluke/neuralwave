package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gofiber/contrib/websocket"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/openfluke/loom/nn"
	"github.com/openfluke/loom/tokenizer"
)

// Data models
type ModelStatus struct {
	Name        string    `json:"name"`
	Path        string    `json:"path,omitempty"`
	Loadable    bool      `json:"loadable"`
	Checked     bool      `json:"checked"`
	LastChecked time.Time `json:"last_checked,omitempty"`
	Error       string    `json:"error,omitempty"`

	// Selection
	Selected bool `json:"selected"`

	// Analysis Data
	Analysis      *ModelAnalysis `json:"analysis,omitempty"`
	BlankAnalysis *ModelAnalysis `json:"blank_analysis,omitempty"`

	// Benchmark Data
	Benchmarked     bool    `json:"benchmarked"`
	TPS             float64 `json:"tps,omitempty"`
	ExampleOutput   string  `json:"example_output,omitempty"`
	GeneratedTokens []int   `json:"generated_tokens,omitempty"`
	HiddenSize      int     `json:"hidden_size,omitempty"`
	VocabSize       int     `json:"vocab_size,omitempty"`
	Params          string  `json:"params,omitempty"`
}

type ModelAnalysis struct {
	TotalParams int64       `json:"total_params"`
	Layers      []LayerInfo `json:"layers"`
}

type LayerInfo struct {
	Name   string  `json:"name"`
	Type   string  `json:"type"`   // e.g. "Attention", "MLP", "Norm"
	Params int64   `json:"params"` // Shape product
	Shape  []int   `json:"shape"`
	Mean   float32 `json:"mean,omitempty"`
	StdDev float32 `json:"std_dev,omitempty"`
	Min    float32 `json:"min,omitempty"`
	Max    float32 `json:"max,omitempty"`
}

type Store struct {
	Models map[string]*ModelStatus `json:"models"`
	mu     sync.RWMutex
}

// LoomModel represents a LOOM AI model with telemetry data
type LoomModel struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Path      string         `json:"path"`
	Telemetry *LoomTelemetry `json:"telemetry"`
}

// LoomTelemetry represents a LOOM network's structure (matches telemetry.go format)
type LoomTelemetry struct {
	ID          string               `json:"id"`
	TotalLayers int                  `json:"total_layers"`
	TotalParams int                  `json:"total_parameters"`
	Layers      []LoomLayerTelemetry `json:"layers"`
}

// LoomLayerTelemetry contains metadata about a specific LOOM layer
type LoomLayerTelemetry struct {
	// Grid position
	GridRow   int `json:"grid_row"`
	GridCol   int `json:"grid_col"`
	CellLayer int `json:"cell_layer"`

	// Layer info
	Type       string `json:"type"`
	Activation string `json:"activation,omitempty"`
	Parameters int    `json:"parameters"`

	// Dimensions
	InputShape  []int `json:"input_shape,omitempty"`
	OutputShape []int `json:"output_shape,omitempty"`

	// For nested/parallel layers
	Branches    []LoomLayerTelemetry `json:"branches,omitempty"`
	CombineMode string               `json:"combine_mode,omitempty"` // "concat", "add", "avg", "grid_scatter"
}

var (
	store = Store{
		Models: make(map[string]*ModelStatus),
	}
	storePath = "static/models.json"

	// WebSocket Hub
	clients   = make(map[*websocket.Conn]bool)
	clientsMu sync.Mutex
	broadcast = make(chan []byte)

	// LOOM Models
	loomModels   = make(map[string]*LoomModel)
	loomModelsMu sync.RWMutex
)

func runWeb(port int) {
	// Ensure static dir exists
	os.MkdirAll("static", 0755)
	loadStore()

	app := fiber.New(fiber.Config{
		AppName:               "NeuralWave Dashboard",
		DisableStartupMessage: true,
	})

	app.Use(logger.New())
	app.Use(cors.New())

	app.Static("/static", "./static")

	// Routes
	app.Get("/", handleDashboard)

	// Upgrade to WebSocket
	app.Use("/ws", func(c *fiber.Ctx) error {
		if websocket.IsWebSocketUpgrade(c) {
			c.Locals("allowed", true)
			return c.Next()
		}
		return c.SendStatus(fiber.StatusUpgradeRequired)
	})
	app.Get("/ws", websocket.New(handleWebSocket))

	// Start Broadcaster
	go runBroadcaster()

	fmt.Printf("\n🌊 NeuralWave Web Interface running at http://localhost:%d\n", port)
	log.Fatal(app.Listen(fmt.Sprintf(":%d", port)))
}

// -- Handlers --

func handleDashboard(c *fiber.Ctx) error {
	content, err := os.ReadFile("views/index.html")
	if err != nil {
		return c.Status(500).SendString("Template not found. Please create views/index.html")
	}
	c.Set("Content-Type", "text/html")
	return c.Send(content)
}

func handleWebSocket(c *websocket.Conn) {
	// Register client
	clientsMu.Lock()
	clients[c] = true
	clientsMu.Unlock()

	defer func() {
		clientsMu.Lock()
		delete(clients, c)
		clientsMu.Unlock()
		c.Close()
	}()

	// Send initial state immediately
	sendState(c)

	type Message struct {
		Action  string          `json:"action"`
		Payload json.RawMessage `json:"payload,omitempty"`
	}

	for {
		var msg Message
		err := c.ReadJSON(&msg)
		if err != nil {
			break
		}

		switch msg.Action {
		case "scan":
			go scanModels()
		case "verify":
			go verifyModels()
		case "benchmark":
			go benchmarkModels()
		case "analyze":
			go analyzeSelectedModels(false)
		case "compare":
			go analyzeSelectedModels(true)
		case "refresh":
			sendState(c)
		case "toggle_select":
			var payload struct {
				Name string `json:"name"`
			}
			if err := json.Unmarshal(msg.Payload, &payload); err == nil && payload.Name != "" {
				toggleSelect(payload.Name)
			}
		case "inspect":
			var payload struct {
				Name string `json:"name"`
			}
			if err := json.Unmarshal(msg.Payload, &payload); err == nil && payload.Name != "" {
				result := inspectModel(payload.Name)
				data, _ := json.Marshal(map[string]interface{}{
					"type": "inspect_result",
					"data": result,
				})
				c.WriteMessage(websocket.TextMessage, data)
			}
		case "scan_loom_folder":
			var payload struct {
				Path string `json:"path"`
			}
			if err := json.Unmarshal(msg.Payload, &payload); err == nil && payload.Path != "" {
				go scanLoomFolder(payload.Path)
			}
		}
	}
}

func runBroadcaster() {
	for msg := range broadcast {
		clientsMu.Lock()
		for client := range clients {
			if err := client.WriteMessage(websocket.TextMessage, msg); err != nil {
				client.Close()
				delete(clients, client)
			}
		}
		clientsMu.Unlock()
	}
}

func broadcastUpdate() {
	store.mu.RLock()
	data, _ := json.Marshal(map[string]interface{}{
		"type": "update",
		"data": store.Models,
	})
	store.mu.RUnlock()
	broadcast <- data
}

func sendState(c *websocket.Conn) {
	store.mu.RLock()
	data, _ := json.Marshal(map[string]interface{}{
		"type": "update",
		"data": store.Models,
	})
	store.mu.RUnlock()
	c.WriteMessage(websocket.TextMessage, data)
}

// -- Logic --

func scanModels() {
	homeDir, _ := os.UserHomeDir()
	hubDir := filepath.Join(homeDir, ".cache", "huggingface", "hub")

	entries, err := os.ReadDir(hubDir)
	if err != nil {
		return
	}

	store.mu.Lock()
	changed := false
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
			rawName := entry.Name()
			modelName := strings.TrimPrefix(rawName, "models--")
			modelName = strings.Replace(modelName, "--", "/", 1)

			modelDir := filepath.Join(hubDir, rawName, "snapshots")
			snaps, _ := os.ReadDir(modelDir)
			if len(snaps) > 0 {
				fullPath := filepath.Join(modelDir, snaps[0].Name())
				if _, exists := store.Models[modelName]; !exists {
					store.Models[modelName] = &ModelStatus{Name: modelName, Path: fullPath}
					changed = true
				} else {
					if store.Models[modelName].Path != fullPath {
						store.Models[modelName].Path = fullPath
						changed = true
					}
				}
			}
		}
	}
	saveStoreLocked() // Helper assumed to be same
	store.mu.Unlock()

	if changed {
		broadcastUpdate()
	}
}

func verifyModels() {
	store.mu.RLock()
	names := make([]string, 0, len(store.Models))
	for name := range store.Models {
		names = append(names, name)
	}
	store.mu.RUnlock()

	for _, name := range names {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("⚠️ Recovered from panic verifying %s: %v\n", name, r)
					store.mu.Lock()
					if model, ok := store.Models[name]; ok {
						model.Loadable = false
						model.Error = fmt.Sprintf("CRASH: %v", r)
						model.Checked = true
						model.LastChecked = time.Now()
						saveStoreLocked()
					}
					store.mu.Unlock()
					broadcastUpdate()
				}
			}()

			store.mu.RLock()
			model := store.Models[name]
			store.mu.RUnlock()

			if model.Checked && model.Loadable {
				return
			}

			// Update UI that we are working on this model (optional, maybe specific status field later)
			// For now just background process

			err := tryLoad(model.Path)

			store.mu.Lock()
			// Re-fetch to be safe
			if m, ok := store.Models[name]; ok {
				m.Checked = true
				m.LastChecked = time.Now()
				if err != nil {
					m.Loadable = false
					m.Error = err.Error()
				} else {
					m.Loadable = true
					m.Error = ""
				}
				saveStoreLocked()
			}
			store.mu.Unlock()

			broadcastUpdate() // Real-time update per model
		}()
	}
}

func benchmarkModels() {
	store.mu.RLock()
	names := make([]string, 0, len(store.Models))
	for name := range store.Models {
		names = append(names, name)
	}
	store.mu.RUnlock()

	for _, name := range names {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("⚠️ Recovered from panic benchmarking %s: %v\n", name, r)
					store.mu.Lock()
					if model, ok := store.Models[name]; ok {
						model.Error = fmt.Sprintf("CRASH: %v", r)
						saveStoreLocked()
					}
					store.mu.Unlock()
					broadcastUpdate()
				}
			}()

			store.mu.RLock()
			model := store.Models[name]
			store.mu.RUnlock()

			if !model.Loadable || model.Benchmarked {
				return
			}

			res, err := runInference(model.Path)

			store.mu.Lock()
			if m, ok := store.Models[name]; ok {
				m.Benchmarked = true
				if err != nil {
					m.Error = "Benchmark failed: " + err.Error()
				} else {
					m.TPS = res.TPS
					m.ExampleOutput = res.Output
					m.GeneratedTokens = res.Tokens
					m.HiddenSize = res.HiddenSize
					m.VocabSize = res.VocabSize
				}
				saveStoreLocked()
			}
			store.mu.Unlock()

			broadcastUpdate()
		}()
	}
}

// -- Helpers --

func tryLoad(path string) error {
	configPath := filepath.Join(path, "config.json")
	if _, err := os.Stat(configPath); err != nil {
		return fmt.Errorf("missing config.json")
	}
	_, err := nn.LoadTransformerFromSafetensors(path)
	return err
}

type BenchResult struct {
	TPS        float64
	Output     string
	Tokens     []int
	HiddenSize int
	VocabSize  int
}

func runInference(path string) (*BenchResult, error) {
	net, err := nn.LoadTransformerFromSafetensors(path)
	if err != nil {
		return nil, err
	}

	tkPath := filepath.Join(path, "tokenizer.json")
	tk, err := tokenizer.LoadFromFile(tkPath)
	if err != nil {
		return nil, fmt.Errorf("tokenizer error: %v", err)
	}

	weightsPath := filepath.Join(path, "model.safetensors")
	tensors, _ := nn.LoadSafetensors(weightsPath)

	var embeddings []float32
	embedKeys := []string{"model.embed_tokens.weight", "transformer.wte.weight", "embeddings.weight"}
	for _, key := range embedKeys {
		if t, ok := tensors[key]; ok {
			embeddings = t
			break
		}
	}
	if embeddings == nil {
		return nil, fmt.Errorf("no embeddings found")
	}

	var finalNorm []float32
	normKeys := []string{"model.norm.weight", "transformer.ln_f.weight", "norm.weight"}
	for _, key := range normKeys {
		if t, ok := tensors[key]; ok {
			finalNorm = t
			break
		}
	}

	start := time.Now()
	// Benchmark prompt
	inputIDs := tk.Encode("Hello, artificial", false)
	tokens := make([]int, len(inputIDs))
	for i, v := range inputIDs {
		tokens[i] = int(v)
	}
	generated := []int{}

	hiddenSize := net.InputSize
	vocabSize := len(embeddings) / hiddenSize

	for i := 0; i < 10; i++ {
		currInput := append(tokens, generated...)

		inputTensor := make([]float32, len(currInput)*hiddenSize)
		for t, tokenID := range currInput {
			if tokenID >= vocabSize || tokenID < 0 {
				continue
			}
			copy(inputTensor[t*hiddenSize:], embeddings[tokenID*hiddenSize:(tokenID+1)*hiddenSize])
		}

		net.BatchSize = 1
		output, _ := net.ForwardCPU(inputTensor)

		if len(finalNorm) > 0 {
			config := &nn.LayerConfig{Type: nn.LayerRMSNorm, NormSize: hiddenSize, Gamma: finalNorm, Epsilon: 1e-6}
			output = nn.RmsNormForwardCPU(output, nil, config, len(currInput))
		}

		lastIdx := (len(currInput) - 1) * hiddenSize
		lastHidden := output[lastIdx : lastIdx+hiddenSize]

		maxIdx := 0
		maxVal := float32(-1e9)
		for v := 0; v < vocabSize; v++ {
			sum := float32(0)
			for d := 0; d < hiddenSize; d++ {
				sum += lastHidden[d] * embeddings[v*hiddenSize+d]
			}
			if sum > maxVal {
				maxVal = sum
				maxIdx = v
			}
		}
		generated = append(generated, maxIdx)
	}

	duration := time.Since(start)
	tps := 10.0 / duration.Seconds()

	genIDs := make([]uint32, len(generated))
	for i, v := range generated {
		genIDs[i] = uint32(v)
	}
	outputStr := tk.Decode(genIDs, true)

	return &BenchResult{
		TPS:        tps,
		Output:     outputStr,
		Tokens:     generated,
		HiddenSize: hiddenSize,
		VocabSize:  vocabSize,
	}, nil
}

// InspectResult holds diagnostic information about a model
type InspectResult struct {
	ModelName string                 `json:"model_name"`
	ModelPath string                 `json:"model_path"`
	Error     string                 `json:"error,omitempty"`
	Config    map[string]interface{} `json:"config,omitempty"`
	Tensors   []TensorInfo           `json:"tensors"`
	Summary   InspectSummary         `json:"summary"`
}

type TensorInfo struct {
	Name       string `json:"name"`
	Recognized bool   `json:"recognized"`
	LayerType  string `json:"layer_type"`
	Pattern    string `json:"pattern,omitempty"` // Which pattern matched
}

type InspectSummary struct {
	TotalTensors      int  `json:"total_tensors"`
	RecognizedTensors int  `json:"recognized_tensors"`
	UnknownTensors    int  `json:"unknown_tensors"`
	HasConfig         bool `json:"has_config"`
	HasSafetensors    bool `json:"has_safetensors"`
	HasTokenizer      bool `json:"has_tokenizer"`
}

// inspectModel analyzes a model's structure without fully loading it
func inspectModel(name string) *InspectResult {
	store.mu.RLock()
	model, exists := store.Models[name]
	store.mu.RUnlock()

	if !exists {
		return &InspectResult{
			ModelName: name,
			Error:     "Model not found in store",
		}
	}

	result := &InspectResult{
		ModelName: name,
		ModelPath: model.Path,
		Error:     model.Error,
		Tensors:   []TensorInfo{},
	}

	// Check for required files
	configPath := filepath.Join(model.Path, "config.json")
	safetensorsPath := filepath.Join(model.Path, "model.safetensors")
	tokenizerPath := filepath.Join(model.Path, "tokenizer.json")

	if _, err := os.Stat(configPath); err == nil {
		result.Summary.HasConfig = true
		// Load and parse config
		if data, err := os.ReadFile(configPath); err == nil {
			var config map[string]interface{}
			if json.Unmarshal(data, &config) == nil {
				result.Config = config
			}
		}
	}

	if _, err := os.Stat(safetensorsPath); err == nil {
		result.Summary.HasSafetensors = true
	}

	if _, err := os.Stat(tokenizerPath); err == nil {
		result.Summary.HasTokenizer = true
	}

	// Load safetensors to get tensor names
	if result.Summary.HasSafetensors {
		tensors, err := nn.LoadSafetensors(safetensorsPath)
		if err != nil {
			result.Error = fmt.Sprintf("Failed to load safetensors: %v", err)
		} else {
			// Analyze each tensor
			for tensorName := range tensors {
				info := classifyTensor(tensorName)
				result.Tensors = append(result.Tensors, info)
				result.Summary.TotalTensors++
				if info.Recognized {
					result.Summary.RecognizedTensors++
				} else {
					result.Summary.UnknownTensors++
				}
			}
			// Sort tensors by name
			sort.Slice(result.Tensors, func(i, j int) bool {
				return naturalLess(result.Tensors[i].Name, result.Tensors[j].Name)
			})
		}
	}

	return result
}

// classifyTensor determines if a tensor name matches known patterns
func classifyTensor(name string) TensorInfo {
	lower := strings.ToLower(name)

	// Known patterns for LOOM-supported architectures
	patterns := []struct {
		pattern   string
		layerType string
	}{
		// Embeddings
		{"embed_tokens.weight", "Embedding"},
		{"wte.weight", "Embedding"},
		{"word_embeddings.weight", "Embedding"},
		{"embeddings.weight", "Embedding"},

		// Attention
		{"self_attn.q_proj", "Attention.Q"},
		{"self_attn.k_proj", "Attention.K"},
		{"self_attn.v_proj", "Attention.V"},
		{"self_attn.o_proj", "Attention.O"},
		{"attention.query", "Attention.Q"},
		{"attention.key", "Attention.K"},
		{"attention.value", "Attention.V"},
		{"attention.output", "Attention.O"},
		{"attn.c_attn", "Attention.QKV"},
		{"attn.c_proj", "Attention.O"},
		{"attn.q_proj", "Attention.Q"},
		{"attn.k_proj", "Attention.K"},
		{"attn.v_proj", "Attention.V"},
		{"attn.o_proj", "Attention.O"},

		// MLP/FFN
		{"mlp.gate_proj", "MLP.Gate"},
		{"mlp.up_proj", "MLP.Up"},
		{"mlp.down_proj", "MLP.Down"},
		{"mlp.c_fc", "MLP.FC"},
		{"mlp.c_proj", "MLP.Proj"},
		{"feed_forward", "MLP"},
		{"ffn", "MLP"},

		// Normalization
		{"input_layernorm", "Norm.Input"},
		{"post_attention_layernorm", "Norm.PostAttn"},
		{"ln_1", "Norm.Pre"},
		{"ln_2", "Norm.Post"},
		{"ln_f", "Norm.Final"},
		{"norm.weight", "Norm"},
		{"layernorm", "Norm"},
		{"rms_norm", "RMSNorm"},

		// Output
		{"lm_head", "LMHead"},

		// Rotary embeddings
		{"rotary_emb", "RotaryEmb"},
		{"inv_freq", "RotaryEmb"},
	}

	for _, p := range patterns {
		if strings.Contains(lower, p.pattern) {
			return TensorInfo{
				Name:       name,
				Recognized: true,
				LayerType:  p.layerType,
				Pattern:    p.pattern,
			}
		}
	}

	// Unknown tensor
	return TensorInfo{
		Name:       name,
		Recognized: false,
		LayerType:  "Unknown",
		Pattern:    "",
	}
}

func loadStore() {
	data, err := os.ReadFile(storePath)
	if err == nil {
		json.Unmarshal(data, &store)
	}
	if store.Models == nil {
		store.Models = make(map[string]*ModelStatus)
	}
	// Enforce sort on load
	for _, m := range store.Models {
		if m.Analysis != nil {
			sort.Slice(m.Analysis.Layers, func(i, j int) bool {
				return naturalLess(m.Analysis.Layers[i].Name, m.Analysis.Layers[j].Name)
			})
		}
		if m.BlankAnalysis != nil {
			sort.Slice(m.BlankAnalysis.Layers, func(i, j int) bool {
				return naturalLess(m.BlankAnalysis.Layers[i].Name, m.BlankAnalysis.Layers[j].Name)
			})
		}
	}
}

func saveStoreLocked() {
	data, _ := json.MarshalIndent(store, "", "  ")
	os.WriteFile(storePath, data, 0644)
}

func toggleSelect(name string) {
	store.mu.Lock()
	if model, ok := store.Models[name]; ok {
		model.Selected = !model.Selected
		saveStoreLocked()
	}
	store.mu.Unlock()
	broadcastUpdate()
}

// simulateBlankAnalysis generates a theoretical "freshly initialized" version of the model
// using simplistic Xavier/Kaiming assumptions based on layer types.
func simulateBlankAnalysis(real *ModelAnalysis) *ModelAnalysis {
	blank := &ModelAnalysis{
		TotalParams: real.TotalParams,
		Layers:      make([]LayerInfo, len(real.Layers)),
	}

	for i, l := range real.Layers {
		// Default: Xavier Init (Gaussian with mean 0, std = sqrt(2 / (fan_in + fan_out)))
		// Since we don't have exact dimensions, we approximate fan_in/out using Params.
		// Approximating square matrix: Dim ~ Sqrt(Params)
		dim := float64(math.Sqrt(float64(l.Params)))
		if dim < 1 {
			dim = 1
		}

		var std float64
		if l.Type == "Norm" || l.Type == "RMSNorm" {
			// Norm layers usually init to 1.0 (gamma) or 0.0 (beta)
			// Let's assume gamma (weights) = 1.0
			blank.Layers[i] = LayerInfo{
				Name:   l.Name,
				Type:   l.Type,
				Params: l.Params,
				Shape:  l.Shape,
				Mean:   1.0,
				StdDev: 0.0,
				Min:    0.9,
				Max:    1.1,
			}
			continue
		} else if l.Type == "Embedding" {
			// Embedding usually N(0, 1) or small uniform
			std = 1.0
		} else {
			// Linear/MLP/Attn: He/Kaiming initialization = sqrt(2/fan_in)
			std = math.Sqrt(2.0 / dim)
		}

		// Projected stats for Gaussian(0, std)
		blank.Layers[i] = LayerInfo{
			Name:   l.Name,
			Type:   l.Type,
			Params: l.Params,
			Shape:  l.Shape, // Copied
			Mean:   0.0,
			StdDev: float32(std),
			Min:    float32(-3 * std),
			Max:    float32(3 * std),
		}
	}
	return blank
}

func analyzeSelectedModels(deep bool) {
	store.mu.RLock()
	names := make([]string, 0, len(store.Models))
	for name := range store.Models {
		names = append(names, name)
	}
	store.mu.RUnlock()

	for _, name := range names {
		// Panic recovery for safety
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("⚠️ Panic analyzing %s: %v\n", name, r)
				}
			}()

			store.mu.RLock()
			model := store.Models[name]
			store.mu.RUnlock()

			if !model.Selected {
				return
			}

			// If already analyzed, maybe we just want to re-sort?
			// But for now, let's allow re-analysis to ensure data consistency if user clicks 'Analyze'.
			// This fixes the 'Processing' hang if everything was skipped.
			fmt.Printf("Analyzing %s...\n", name)

			// Load weights header only (fast)
			path := filepath.Join(model.Path, "model.safetensors")
			tensors, err := nn.LoadSafetensors(path) // This loads data too? Wait, nn.LoadSafetensors in loom might load data.
			// Actually nn.LoadSafetensors returns map[string][]float32. That loads EVERYTHING into RAM.
			// Ideally we want metadata only. But for now, we risk it or assume "small models" fit in RAM.
			// Given user constraints ("small models"), loading one by one is acceptable.

			if err != nil {
				fmt.Printf("Error loading %s: %v\n", name, err)
				return
			}

			analysis := &ModelAnalysis{
				Layers: []LayerInfo{},
			}

			var totalParams int64
			for key, data := range tensors {
				count := int64(len(data))
				totalParams += count

				// Simple heuristic for type
				lType := "Param"
				lower := strings.ToLower(key)
				if strings.Contains(lower, "attn") || strings.Contains(lower, "attention") {
					lType = "Attention"
				} else if strings.Contains(lower, "mlp") || strings.Contains(lower, "feedforward") {
					lType = "MLP"
				} else if strings.Contains(lower, "norm") {
					lType = "Norm"
				} else if strings.Contains(lower, "embed") {
					lType = "Embedding"
				}

				var mean, std, min, max float32
				if deep {
					mean, std, min, max = calcStats(data)
				}

				// Update or append layer info
				// Note: If we run non-deep then deep, we might overwrite?
				// The prompt implies we want distinct buttons.
				// If deep=false (Analyze), we just want basics.
				// If deep=true (Compare), we want stats.
				// Let's just overwrite for simplicity, assuming user knows.

				analysis.Layers = append(analysis.Layers, LayerInfo{
					Name:   key,
					Type:   lType,
					Params: count,
					Shape:  []int{len(data)},
					Mean:   mean,
					StdDev: std,
					Min:    min,
					Max:    max,
				})
			}

			// Clean up heavy memory immediately
			tensors = nil

			// Natural Sort Layers
			sort.Slice(analysis.Layers, func(i, j int) bool {
				return naturalLess(analysis.Layers[i].Name, analysis.Layers[j].Name)
			})

			analysis.TotalParams = totalParams

			// Generate Blank Counterpart
			blankAnalysis := simulateBlankAnalysis(analysis)

			store.mu.Lock()
			if m, ok := store.Models[name]; ok {
				m.Analysis = analysis
				m.BlankAnalysis = blankAnalysis
				saveStoreLocked()
			}
			store.mu.Unlock()
			broadcastUpdate()
		}()
	}
}

// naturalLess compares two strings with natural number ordering
func naturalLess(s1, s2 string) bool {
	re := regexp.MustCompile(`\d+|\D+`)
	parts1 := re.FindAllString(s1, -1)
	parts2 := re.FindAllString(s2, -1)

	for i := 0; i < len(parts1) && i < len(parts2); i++ {
		p1 := parts1[i]
		p2 := parts2[i]

		// Check if both are numbers
		n1, err1 := strconv.Atoi(p1)
		n2, err2 := strconv.Atoi(p2)

		if err1 == nil && err2 == nil {
			if n1 != n2 {
				return n1 < n2
			}
		} else {
			if p1 != p2 {
				return p1 < p2
			}
		}
	}
	return len(parts1) < len(parts2)
}

func calcStats(data []float32) (float32, float32, float32, float32) {
	if len(data) == 0 {
		return 0, 0, 0, 0
	}
	var sum, minVal, maxVal float32
	minVal = data[0]
	maxVal = data[0]

	for _, v := range data {
		sum += v
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	mean := sum / float32(len(data))

	var varianceSum float32
	for _, v := range data {
		diff := v - mean
		varianceSum += diff * diff
	}
	stdDev := float32(math.Sqrt(float64(varianceSum / float32(len(data)))))

	return mean, stdDev, minVal, maxVal
}

// scanLoomFolder scans a directory for LOOM model telemetry files
func scanLoomFolder(folderPath string) {
	fmt.Printf("📂 Scanning LOOM folder: %s\n", folderPath)

	entries, err := os.ReadDir(folderPath)
	if err != nil {
		fmt.Printf("⚠️ Error reading folder: %v\n", err)
		return
	}

	loomModelsMu.Lock()
	// Clear existing models
	loomModels = make(map[string]*LoomModel)

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		// Only process *_telemetry.json files
		if !strings.HasSuffix(name, "_telemetry.json") {
			continue
		}

		filePath := filepath.Join(folderPath, name)
		data, err := os.ReadFile(filePath)
		if err != nil {
			fmt.Printf("⚠️ Error reading %s: %v\n", name, err)
			continue
		}

		var telemetry LoomTelemetry
		if err := json.Unmarshal(data, &telemetry); err != nil {
			fmt.Printf("⚠️ Error parsing %s: %v\n", name, err)
			continue
		}

		// Extract model name from filename (remove _telemetry.json suffix)
		modelName := strings.TrimSuffix(name, "_telemetry.json")
		if telemetry.ID != "" {
			modelName = telemetry.ID
		}

		loomModels[modelName] = &LoomModel{
			ID:        modelName,
			Name:      modelName,
			Path:      filePath,
			Telemetry: &telemetry,
		}
		fmt.Printf("  ✓ Loaded LOOM model: %s (%d layers, %d params)\n",
			modelName, telemetry.TotalLayers, telemetry.TotalParams)
	}
	loomModelsMu.Unlock()

	fmt.Printf("📦 Found %d LOOM models\n", len(loomModels))
	broadcastLoomUpdate()
}

// broadcastLoomUpdate sends the current LOOM models to all connected clients
func broadcastLoomUpdate() {
	loomModelsMu.RLock()
	data, _ := json.Marshal(map[string]interface{}{
		"type": "loom_update",
		"data": loomModels,
	})
	loomModelsMu.RUnlock()
	broadcast <- data
}
