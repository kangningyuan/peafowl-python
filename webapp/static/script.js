let currentSection = 'overview';

function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    document.getElementById(sectionId).classList.add('active');
    
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const buttons = document.querySelectorAll('.nav-btn');
    if (sectionId === 'overview') buttons[0].classList.add('active');
    if (sectionId === 'demo') buttons[1].classList.add('active');
    if (sectionId === 'cryptography') buttons[2].classList.add('active');
    if (sectionId === 'steps') buttons[3].classList.add('active');
    if (sectionId === 'mnist-training') buttons[4].classList.add('active');
    
    currentSection = sectionId;
}

async function runDemo() {
    const numParties = parseInt(document.getElementById('num-parties').value) || 3;
    
    try {
        const response = await fetch('/api/generate_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ num_parties: numParties })
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('demo-result').style.display = 'block';
            
            // 更新统计数据
            document.getElementById('stat-parties').textContent = data.protocol_result.data_providers.length;
            document.getElementById('stat-intersection').textContent = data.protocol_result.server.intersection_size;
            document.getElementById('stat-aligned').textContent = data.protocol_result.total_aligned_samples;
            document.getElementById('stat-features').textContent = data.protocol_result.total_aligned_samples * 
                data.protocol_result.data_providers.reduce((sum, p) => sum + p.aligned_features_shape[1], 0);
            
            // 更新参与方信息
            const partiesContainer = document.getElementById('parties-info');
            partiesContainer.innerHTML = data.protocol_result.data_providers.map(dp => `
                <div class="party-card">
                    <h4>${dp.id}</h4>
                    <div class="info">
                        <div>原始样本数: <strong>${dp.original_samples}</strong></div>
                        <div>对齐样本数: <strong>${dp.aligned_samples}</strong></div>
                        <div>特征维度: <strong>${dp.aligned_features_shape[1]}</strong></div>
                    </div>
                </div>
            `).join('');
            
            // 更新对齐特征
            const featuresContainer = document.getElementById('aligned-features');
            featuresContainer.innerHTML = Object.entries(data.protocol_result.aligned_features).map(([pid, shape]) => `
                <div class="feature-item">
                    <h4>${pid} 对齐特征</h4>
                    <div>形状: ${shape[0]} × ${shape[1]}</div>
                    <div>样本数: ${shape[0]}</div>
                    <div>特征维度: ${shape[1]}</div>
                </div>
            `).join('');
            
            // 更新流程可视化
            updateFlowVisualization(data);
        } else {
            alert('错误: ' + data.error);
        }
    } catch (error) {
        alert('请求失败: ' + error.message);
    }
}

function updateFlowVisualization(data) {
    // 更新各方数据
    data.parties.forEach((party, index) => {
        const flowId = `flow-p${index}`;
        const container = document.getElementById(flowId);
        if (container) {
            container.innerHTML = `
                <div>样本数: ${party.num_samples}</div>
                <div>特征数: ${party.num_features}</div>
                <div>ID示例: ${party.ids.slice(0, 3).join(', ')}...</div>
            `;
        }
    });
    
    // 更新加密阶段
    const encryptContainer = document.getElementById('flow-encrypt');
    if (encryptContainer) {
        encryptContainer.innerHTML = `
            <div>加密方法: HMAC-SHA256</div>
            <div>加密ID数: ${data.parties.reduce((sum, p) => sum + p.num_samples, 0)}</div>
        `;
    }
    
    // 更新服务器
    const serverContainer = document.getElementById('flow-server');
    if (serverContainer) {
        serverContainer.innerHTML = `
            <div>接收方数: ${data.protocol_result.server.encrypted_parties}</div>
            <div>计算交集...</div>
        `;
    }
    
    // 更新交集
    const intersectContainer = document.getElementById('flow-intersect');
    if (intersectContainer) {
        intersectContainer.innerHTML = `
            <div>交集大小: ${data.protocol_result.server.intersection_size}</div>
            <div>交集率: ${(data.protocol_result.server.intersection_size / data.parties[0].num_samples * 100).toFixed(1)}%</div>
        `;
    }
    
    // 更新对齐
    const alignContainer = document.getElementById('flow-align');
    if (alignContainer) {
        alignContainer.innerHTML = `
            <div>对齐样本: ${data.protocol_result.total_aligned_samples}</div>
            <div>总特征: ${data.protocol_result.total_aligned_samples * 
                data.protocol_result.data_providers.reduce((sum, p) => sum + p.aligned_features_shape[1], 0)}</div>
        `;
    }
}

async function showCryptoDetails(cryptoType) {
    try {
        const response = await fetch('/api/cryptography_details', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ type: cryptoType })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const cryptoDetails = document.getElementById('crypto-details');
            document.getElementById('crypto-title').textContent = data.crypto.example ? 
                `${data.crypto.type} - 示例` : data.crypto.type;
            document.getElementById('crypto-description').textContent = data.crypto.description;
            
            let exampleHTML = '';
            if (data.crypto.example) {
                exampleHTML = '<strong>示例:</strong><pre>' + JSON.stringify(data.crypto.example, null, 2) + '</pre>';
            }
            if (data.crypto.parameters) {
                exampleHTML += '<strong>参数:</strong><pre>' + JSON.stringify(data.crypto.parameters, null, 2) + '</pre>';
            }
            
            document.getElementById('crypto-example').innerHTML = exampleHTML;
            cryptoDetails.style.display = 'block';
            
            // 滚动到详情区域
            cryptoDetails.scrollIntoView({ behavior: 'smooth' });
        }
    } catch (error) {
        alert('获取详情失败: ' + error.message);
    }
}

function hideCryptoDetails() {
    document.getElementById('crypto-details').style.display = 'none';
}

async function runStepByStep() {
    const numParties = parseInt(document.getElementById('num-parties').value) || 3;
    
    try {
        const response = await fetch('/api/step_by_step', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ num_parties: numParties })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = data.steps.map(step => `
                <div class="step-item">
                    <div class="step-number">${step.step}</div>
                    <h3>${step.name}</h3>
                    <p>${step.description}</p>
                    <div class="step-data">
                        <strong>详细信息:</strong>
                        <pre>${JSON.stringify(step.data, null, 2)}</pre>
                    </div>
                </div>
            `).join('');
            
            // 滚动到步骤容器
            stepsContainer.scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('错误: ' + data.error);
        }
    } catch (error) {
        alert('请求失败: ' + error.message);
    }
}

async function runMnistTraining() {
    const numParties = parseInt(document.getElementById('mnist-num-parties').value) || 3;
    const modelType = document.getElementById('model-type').value;
    
    try {
        const response = await fetch('/api/train_mnist', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                num_parties: numParties,
                model_type: modelType
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const result = data.training_result;
            document.getElementById('mnist-training-result').style.display = 'block';
            
            document.getElementById('mnist-parties').textContent = result.dataset_info.num_parties;
            document.getElementById('mnist-aligned').textContent = result.alignment_info.aligned_samples;
            document.getElementById('mnist-features').textContent = result.alignment_info.aligned_features;
            document.getElementById('mnist-accuracy').textContent = (result.results.accuracy * 100).toFixed(2) + '%';
            
            document.getElementById('mnist-dataset-info').innerHTML = `
                <div class="info-item">
                    <strong>参与方数量:</strong> ${result.dataset_info.num_parties}
                </div>
                <div class="info-item">
                    <strong>各参与方样本数:</strong> [${result.dataset_info.samples_per_party.join(', ')}]
                </div>
                <div class="info-item">
                    <strong>各参与方特征数:</strong> [${result.dataset_info.features_per_party.join(', ')}]
                </div>
                <div class="info-item">
                    <strong>原始总特征数:</strong> ${result.dataset_info.total_features_before}
                </div>
                <div class="info-item">
                    <strong>对齐后特征数:</strong> ${result.dataset_info.total_features_after}
                </div>
                <div class="info-item">
                    <strong>特征压缩率:</strong> ${result.dataset_info.compression_rate}
                </div>
            `;
            
            let modelInfoHtml = `
                <div class="info-item">
                    <strong>算法:</strong> ${result.model_info.algorithm}
                </div>
                <div class="info-item">
                    <strong>训练样本数:</strong> ${result.model_info.train_samples}
                </div>
                <div class="info-item">
                    <strong>测试样本数:</strong> ${result.model_info.test_samples}
                </div>
            `;
            
            if (result.model_info.architecture) {
                modelInfoHtml += `
                <div class="info-item">
                    <strong>网络架构:</strong> ${result.model_info.architecture}
                </div>
                `;
            }
            
            if (result.model_info.num_classes) {
                modelInfoHtml += `
                <div class="info-item">
                    <strong>类别数:</strong> ${result.model_info.num_classes}
                </div>
                `;
            }
            
            document.getElementById('mnist-model-performance').innerHTML = modelInfoHtml;
            
            let reportHtml = '';
            if (result.results.loss !== undefined) {
                reportHtml = `
                    <div class="info-item">
                        <strong>测试损失:</strong> ${result.results.loss.toFixed(4)}
                    </div>
                    <div class="info-item">
                        <strong>准确率:</strong> ${(result.results.accuracy * 100).toFixed(2)}%
                    </div>
                    <div class="info-item">
                        <strong>正确分类数:</strong> ${result.results.correct_count}
                    </div>
                    <div class="info-item">
                        <strong>错误分类数:</strong> ${result.results.incorrect_count}
                    </div>
                `;
            } else if (result.results.classification_report) {
                const report = result.results.classification_report;
                reportHtml = `
                    <div class="info-item">
                        <strong>准确率 (Accuracy):</strong> ${result.results.accuracy.toFixed(4)}
                    </div>
                    <div class="info-item">
                        <strong>精确率 (Precision):</strong> ${report.precision.toFixed(4)}
                    </div>
                    <div class="info-item">
                        <strong>召回率 (Recall):</strong> ${report.recall.toFixed(4)}
                    </div>
                    <div class="info-item">
                        <strong>F1分数:</strong> ${report['f1-score'].toFixed(4)}
                    </div>
                `;
            }
            document.getElementById('mnist-classification-report').innerHTML = reportHtml;
            
            if (result.visualizations) {
                renderVisualizations(result.visualizations.correct, 'correct-visualizations', true);
                renderVisualizations(result.visualizations.incorrect, 'incorrect-visualizations', false);
            }
            
            document.getElementById('mnist-training').scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('训练失败: ' + data.error);
        }
    } catch (error) {
        alert('请求失败: ' + error.message);
    }
}

function renderVisualizations(visualizations, containerId, isCorrect) {
    const container = document.getElementById(containerId);
    
    if (!visualizations || visualizations.length === 0) {
        container.innerHTML = '<p class="no-data">暂无数据</p>';
        return;
    }
    
    container.innerHTML = visualizations.map((item, index) => {
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        const ctx = canvas.getContext('2d');
        
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const value = Math.floor(item.image[y][x] * 255);
                ctx.fillStyle = `rgb(${255 - value}, ${255 - value}, ${255 - value})`;
                ctx.fillRect(x, y, 1, 1);
            }
        }
        
        return `
            <div class="digit-card ${isCorrect ? 'correct' : 'incorrect'}">
                <div class="digit-image">
                    <img src="${canvas.toDataURL()}" alt="digit" width="56" height="56">
                </div>
                <div class="digit-info">
                    <div class="digit-label">
                        <span class="label-text">真实:</span>
                        <span class="label-value">${item.true_label}</span>
                    </div>
                    <div class="digit-label">
                        <span class="label-text">预测:</span>
                        <span class="label-value ${isCorrect ? '' : 'wrong'}">${item.predicted_label}</span>
                    </div>
                    <div class="digit-confidence">
                        置信度: ${(item.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    const tempCanvases = container.querySelectorAll('canvas');
    tempCanvases.forEach((canvas, index) => {
        if (visualizations[index]) {
            const ctx = canvas.getContext('2d');
            const item = visualizations[index];
            for (let y = 0; y < 28; y++) {
                for (let x = 0; x < 28; x++) {
                    const value = Math.floor(item.image[y][x] * 255);
                    ctx.fillStyle = `rgb(${255 - value}, ${255 - value}, ${255 - value})`;
                    ctx.fillRect(x, y, 1, 1);
                }
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    showSection('overview');
});
