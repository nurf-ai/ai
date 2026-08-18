package ai

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

const traceLevel = zapcore.Level(-2)

var logger = zap.NewNop()

func SetLogger(l *zap.Logger) {
	if l != nil {
		logger = l
	}
}
