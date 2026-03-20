<script lang="ts">
	import { getSpatialDims, formatValue } from './tensorUtils';
	import { rangeScroll } from './rangeScroll';
	import { VoxelRenderer } from './webgpu/VoxelRenderer';

	let { main, ref, shape, mainLabel = 'Main', refLabel = 'Reference' }: {
		main: Float32Array;
		ref: Float32Array;
		shape: number[];
		mainLabel?: string;
		refLabel?: string;
	} = $props();

	let gpuCanvas: HTMLCanvasElement | undefined = $state();
	let overlayCanvas: HTMLCanvasElement | undefined = $state();
	let container: HTMLDivElement | undefined = $state();
	let containerW = $state(800);
	let containerH = $state(600);
	let rotX = $state(-25);
	let rotY = $state(30);
	let threshold = $state(0.30);
	let opacitySlider = $state(0.77); // quadratic: 0.77² ≈ 0.60
	let opacity = $derived(opacitySlider * opacitySlider);
	let alphaPower = $state(0.5);

	let dragging = $state(false);
	let dragStartX = $state(0);
	let dragStartY = $state(0);
	let rotStartX = $state(0);
	let rotStartY = $state(0);

	let userZoom = $state(1.0);

	// World-space pivot point (maps to screen center)
	let pivotX = $state(0);
	let pivotY = $state(0);
	let pivotZ = $state(0);
	let panning = $state(false);
	let panStartX = $state(0);
	let panStartY = $state(0);
	let panOriginPivotX = $state(0);
	let panOriginPivotY = $state(0);
	let panOriginPivotZ = $state(0);
	// Snapshot of rotation/scale at pan-drag start
	let panCosX = $state(1);
	let panSinX = $state(0);
	let panCosY = $state(1);
	let panSinY = $state(0);
	let panScale = $state(1);

	// Axis slice ranges (in original tensor coords)
	let sliceC = $state<[number, number]>([0, 0]);
	let sliceH = $state<[number, number]>([0, 0]);
	let sliceW = $state<[number, number]>([0, 0]);

	// Reset slice ranges only when the actual tensor shape changes (not chunk size)
	$effect(() => {
		const { channels: origC, height: origH, width: origW } = getSpatialDims(shape);
		sliceC = [0, origC - 1];
		sliceH = [0, origH - 1];
		sliceW = [0, origW - 1];
	});

	// Tooltip state
	let tooltip = $state<{ x: number; y: number; ci: number; iy: number; ix: number; mainVal: number; refVal: number; diffVal: number } | null>(null);
	// Hovered voxel in downsampled grid coords (for outline rendering)
	let hoveredVoxel = $state<{ ci: number; yi: number; xi: number } | null>(null);

	// Render params shared between render effect and mousemove
	let renderParams = $state<{
		scale: number; cosX: number; sinX: number; cosY: number; sinY: number;
		centerX: number; centerY: number;
		C: number; H: number; W: number; depthExtent: number;
		dsC: number; dsH: number; dsW: number;
	} | null>(null);

	// ResizeObserver for container sizing
	$effect(() => {
		if (!container) return;
		const obs = new ResizeObserver((entries) => {
			for (const entry of entries) {
				containerW = entry.contentRect.width;
				containerH = entry.contentRect.height;
			}
		});
		obs.observe(container);
		return () => obs.disconnect();
	});

	// Color schemes — each is an array of RGB stops interpolated across [0,1]
	const COLOR_SCHEMES: Record<string, [number, number, number][]> = {
		plasma: [
			[70, 10, 150],
			[170, 30, 180],
			[230, 60, 120],
			[250, 130, 50],
			[250, 210, 60],
			[240, 250, 130],
		],
		turbo: [
			[50, 60, 200],
			[30, 180, 220],
			[60, 230, 130],
			[200, 230, 50],
			[250, 160, 40],
			[250, 70, 40],
		],
		viridis: [
			[68, 1, 84],
			[60, 82, 138],
			[33, 145, 140],
			[53, 183, 121],
			[143, 215, 68],
			[253, 231, 37],
		],
		magma: [
			[0, 0, 0],
			[80, 18, 123],
			[182, 55, 100],
			[230, 107, 57],
			[252, 194, 68],
			[252, 253, 191],
		],
		inferno: [
			[30, 10, 60],
			[120, 28, 130],
			[200, 50, 80],
			[240, 120, 30],
			[250, 200, 50],
			[255, 255, 160],
		],
		cool: [
			[80, 160, 255],
			[120, 200, 255],
			[160, 240, 220],
			[200, 255, 180],
			[240, 240, 140],
			[255, 255, 200],
		],
	};

	let colorScheme = $state<string>('plasma');

	function sampleColormap(t: number): [number, number, number] {
		const stops = COLOR_SCHEMES[colorScheme] ?? COLOR_SCHEMES.plasma;
		const clamped = Math.max(0, Math.min(1, t));
		const segment = clamped * (stops.length - 1);
		const idx = Math.min(Math.floor(segment), stops.length - 2);
		const local = segment - idx;
		const a = stops[idx];
		const b = stops[idx + 1];
		return [
			Math.round(a[0] + (b[0] - a[0]) * local),
			Math.round(a[1] + (b[1] - a[1]) * local),
			Math.round(a[2] + (b[2] - a[2]) * local),
		];
	}

	const MAX_VOXELS = 50000;

	// User-controlled chunk size
	let chunkSize = $state(3);

	let volumeData = $derived.by(() => {
		const dims = getSpatialDims(shape);
		const { channels: C, height: H, width: W } = dims;

		const diff = new Float32Array(C * H * W);
		let minVal = Infinity;
		let maxVal = -Infinity;

		for (let i = 0; i < C * H * W; i++) {
			const v = Math.abs(main[i] - ref[i]);
			diff[i] = v;
			if (v < minVal) minVal = v;
			if (v > maxVal) maxVal = v;
		}

		if (maxVal === minVal) maxVal = minVal + 1;

		const factor = chunkSize;

		// No downsampling needed
		if (factor <= 1) {
			return { diff, C, H, W, minVal, maxVal, dsC: 1, dsH: 1, dsW: 1, dsOrigin: null, origC: C, origH: H, origW: W };
		}
		const nC = Math.max(1, Math.ceil(C / factor));
		const nH = Math.max(1, Math.ceil(H / factor));
		const nW = Math.max(1, Math.ceil(W / factor));
		const dsSize = nC * nH * nW;
		const dsDiff = new Float32Array(dsSize);
		// Track which original voxel had the max in each block
		const dsOrigin = new Uint32Array(dsSize * 3); // [origC, origH, origW] per block

		for (let ci = 0; ci < nC; ci++) {
			for (let yi = 0; yi < nH; yi++) {
				for (let xi = 0; xi < nW; xi++) {
					let mx = 0, mxC = ci * factor, mxY = yi * factor, mxW = xi * factor;
					for (let dc = 0; dc < factor && ci * factor + dc < C; dc++) {
						for (let dy = 0; dy < factor && yi * factor + dy < H; dy++) {
							for (let dx = 0; dx < factor && xi * factor + dx < W; dx++) {
								const v = diff[(ci * factor + dc) * H * W + (yi * factor + dy) * W + (xi * factor + dx)];
								if (v > mx) {
									mx = v;
									mxC = ci * factor + dc;
									mxY = yi * factor + dy;
									mxW = xi * factor + dx;
								}
							}
						}
					}
					const idx = ci * nH * nW + yi * nW + xi;
					dsDiff[idx] = mx;
					dsOrigin[idx * 3] = mxC;
					dsOrigin[idx * 3 + 1] = mxY;
					dsOrigin[idx * 3 + 2] = mxW;
				}
			}
		}

		return { diff: dsDiff, C: nC, H: nH, W: nW, minVal, maxVal, dsC: factor, dsH: factor, dsW: factor, dsOrigin, origC: C, origH: H, origW: W };
	});

	function onMouseDown(e: MouseEvent) {
		hoveredVoxel = null;
		tooltip = null;
		// Middle-click or shift+left-click → pan
		if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
			e.preventDefault();
			panning = true;
			panStartX = e.clientX;
			panStartY = e.clientY;
			panOriginPivotX = pivotX;
			panOriginPivotY = pivotY;
			panOriginPivotZ = pivotZ;
			const rp = renderParams;
			if (rp) {
				panCosX = rp.cosX; panSinX = rp.sinX;
				panCosY = rp.cosY; panSinY = rp.sinY;
				panScale = rp.scale;
			}
			return;
		}
		dragging = true;
		dragStartX = e.clientX;
		dragStartY = e.clientY;
		rotStartX = rotX;
		rotStartY = rotY;
	}

	function onMouseMove(e: MouseEvent) {
		if (panning) {
			// Inverse projection: screen delta → world-space pivot delta (zero depth component)
			const sdx = e.clientX - panStartX;
			const sdy = e.clientY - panStartY;
			pivotX = panOriginPivotX + (-sdx * panCosY + sdy * panSinX * panSinY) / panScale;
			pivotY = panOriginPivotY + (-sdy * panCosX) / panScale;
			pivotZ = panOriginPivotZ + (sdx * panSinY + sdy * panSinX * panCosY) / panScale;
			// Clamp pivot to volume bounding box
			const { C, H, W, dsC, dsH, dsW } = volumeData;
			const halfW = W * dsW / 2;
			const halfH = H * dsH / 2;
			const origC2 = C * dsC;
			const halfD = origC2 > 1 ? origC2 / 2 : 0;
			pivotX = Math.max(-halfW, Math.min(halfW, pivotX));
			pivotY = Math.max(-halfH, Math.min(halfH, pivotY));
			pivotZ = Math.max(-halfD, Math.min(halfD, pivotZ));
			hoveredVoxel = null;
			tooltip = null;
			return;
		}
		if (dragging) {
			rotY = rotStartY + (e.clientX - dragStartX) * 0.5;
			rotX = rotStartX + (e.clientY - dragStartY) * 0.5;
			rotX = Math.max(-89, Math.min(89, rotX));
			hoveredVoxel = null;
			tooltip = null;
			return;
		}

		// Hit-test for tooltip — project each voxel center, find closest to mouse
		if (!gpuCanvas || !renderParams) return;
		const rect = gpuCanvas.getBoundingClientRect();
		const mouseX = e.clientX - rect.left;
		const mouseY = e.clientY - rect.top;

		const rp = renderParams;
		const { C, H, W, scale, cosY, sinY, cosX, sinX, centerX, centerY, depthExtent, dsC, dsH, dsW } = rp;
		const { diff, minVal, maxVal, dsOrigin, origH: oH, origW: oW } = volumeData;
		const span = maxVal - minVal || 1;
		const invSpan = 1 / span;

		const voxW = dsW;
		const voxH = dsH;
		const voxD = C > 1 ? depthExtent / C : 1;

		// Iterate front-to-back (reverse of render order) to find topmost voxel
		const cStart = cosY > 0 ? C - 1 : 0, cEnd = cosY > 0 ? -1 : C, cStep = cosY > 0 ? -1 : 1;
		const hStart = sinX > 0 ? H - 1 : 0, hEnd = sinX > 0 ? -1 : H, hStep = sinX > 0 ? -1 : 1;
		const wStart = sinY > 0 ? 0 : W - 1, wEnd = sinY > 0 ? W : -1, wStep = sinY > 0 ? 1 : -1;

		const hitRadius = Math.max(voxW, voxH, voxD) * scale * 0.6;
		const hitR2 = hitRadius * hitRadius;

		// Convert slice ranges from original coords to downsampled coords for hit-test
		const cMin = Math.floor(sliceC[0] / dsC), cMax = Math.min(C - 1, Math.floor(sliceC[1] / dsC));
		const hMin = Math.floor(sliceH[0] / dsH), hMax = Math.min(H - 1, Math.floor(sliceH[1] / dsH));
		const wMin = Math.floor(sliceW[0] / dsW), wMax = Math.min(W - 1, Math.floor(sliceW[1] / dsW));

		for (let ci = cStart; ci !== cEnd; ci += cStep) {
			if (ci < cMin || ci > cMax) continue;
			for (let yi = hStart; yi !== hEnd; yi += hStep) {
				if (yi < hMin || yi > hMax) continue;
				for (let xi = wStart; xi !== wEnd; xi += wStep) {
					if (xi < wMin || xi > wMax) continue;
					const dsIdx = ci * H * W + yi * W + xi;
					const v = diff[dsIdx];
					const norm = (v - minVal) * invSpan;
					if (norm < threshold) continue;

					const wx = (xi + 0.5) * voxW - W * dsW / 2;
					const wy = (yi + 0.5) * voxH - H * dsH / 2;
					const wz = (ci + 0.5) * voxD - depthExtent / 2;

					const [sx, sy] = project3D(wx, wy, wz, cosX, sinX, cosY, sinY, centerX, centerY, scale);
					const dx = sx - mouseX, dy = sy - mouseY;
					if (dx * dx + dy * dy < hitR2) {
						// Use exact original coordinates from dsOrigin when downsampled
						let origCi: number, origYi: number, origXi: number;
						if (dsOrigin) {
							origCi = dsOrigin[dsIdx * 3];
							origYi = dsOrigin[dsIdx * 3 + 1];
							origXi = dsOrigin[dsIdx * 3 + 2];
						} else {
							origCi = ci;
							origYi = yi;
							origXi = xi;
						}
						const mainIdx = origCi * oH * oW + origYi * oW + origXi;
						hoveredVoxel = { ci, yi, xi };
						tooltip = {
							x: e.clientX, y: e.clientY,
							ci: origCi, iy: origYi, ix: origXi,
							mainVal: main[mainIdx] ?? 0,
							refVal: ref[mainIdx] ?? 0,
							diffVal: v,
						};
						return;
					}
				}
			}
		}
		hoveredVoxel = null;
		tooltip = null;
	}

	function onMouseUp() {
		dragging = false;
		panning = false;
	}

	function onMouseLeave() {
		dragging = false;
		panning = false;
		hoveredVoxel = null;
		tooltip = null;
	}

	function onWheel(e: WheelEvent) {
		e.preventDefault();
		userZoom *= e.deltaY > 0 ? 0.9 : 1.1;
		userZoom = Math.max(0.1, Math.min(10, userZoom));
	}

	// 3D projection helper: rotate point and project to 2D
	function project3D(
		x: number, y: number, z: number,
		cosX: number, sinX: number, cosY: number, sinY: number,
		centerX: number, centerY: number, scale: number,
	): [number, number] {
		// Rotate around Y axis
		const rx = x * cosY - z * sinY;
		const rz = x * sinY + z * cosY;
		// Rotate around X axis
		const ry = y * cosX - rz * sinX;
		return [centerX + rx * scale, centerY + ry * scale];
	}

	// --- WebGPU renderer lifecycle ---
	let renderer: VoxelRenderer | null = $state(null);

	// Init/destroy renderer when canvas mounts
	$effect(() => {
		if (!gpuCanvas) return;
		let cancelled = false;
		VoxelRenderer.create(gpuCanvas).then((r) => {
			if (cancelled) {
				r?.destroy();
				return;
			}
			renderer = r;
		});
		return () => {
			cancelled = true;
			renderer?.destroy();
			renderer = null;
		};
	});

	// Upload volume data when it changes
	$effect(() => {
		if (!renderer) return;
		const { diff, C, H, W, minVal, maxVal, dsC, dsH, dsW } = volumeData;
		const origC = C * dsC;
		const depthExtent = origC > 1 ? origC : 0;
		const voxW = dsW;
		const voxH = dsH;
		const voxD = origC > 1 ? dsC : 1;
		renderer.setVolumeData(diff, C, H, W, minVal, maxVal, voxW, voxH, voxD);
	});

	// Update camera when rotation/zoom/pivot/size changes
	$effect(() => {
		if (!renderer) return;
		const { C, H, W, dsC, dsH, dsW } = volumeData;
		const _rotX = rotX;
		const _rotY = rotY;
		const _userZoom = userZoom;
		const _pivotX = pivotX;
		const _pivotY = pivotY;
		const _pivotZ = pivotZ;
		const _cw = containerW;
		const _ch = containerH;

		const origW2 = W * dsW;
		const origH2 = H * dsH;
		const origC2 = C * dsC;
		const depthExtent = origC2 > 1 ? origC2 : 0;

		const diagonal = Math.sqrt(origW2 * origW2 + origH2 * origH2 + depthExtent * depthExtent) || 1;
		const margin = 80;
		const cw = Math.max(1, Math.floor(_cw));
		const ch = Math.max(1, Math.floor(_ch));
		const baseScale = Math.min((cw - margin) / diagonal, (ch - margin) / diagonal);
		const scale = baseScale * _userZoom;

		// Compute renderParams for CPU hit-testing
		const cosX = Math.cos(_rotX * Math.PI / 180);
		const sinX = Math.sin(_rotX * Math.PI / 180);
		const cosY = Math.cos(_rotY * Math.PI / 180);
		const sinY = Math.sin(_rotY * Math.PI / 180);

		const rpx = _pivotX * cosY - _pivotZ * sinY;
		const rpz = _pivotX * sinY + _pivotZ * cosY;
		const rpy = _pivotY * cosX - rpz * sinX;
		const centerX = cw / 2 - rpx * scale;
		const centerY = ch / 2 - rpy * scale;

		renderParams = { scale, cosX, sinX, cosY, sinY, centerX, centerY, C, H, W, depthExtent, dsC, dsH, dsW };

		renderer.setCamera(_rotX, _rotY, scale, _pivotX, _pivotY, _pivotZ, cw, ch);
	});

	// Update filter uniforms when threshold/opacity/slices change
	$effect(() => {
		if (!renderer) return;
		const { C, H, W, minVal, maxVal, dsC, dsH, dsW } = volumeData;
		const _threshold = threshold;
		const _opacity = opacity;
		const _sliceC = sliceC;
		const _sliceH = sliceH;
		const _sliceW = sliceW;
		const _alphaPower = alphaPower;

		const origW2 = W * dsW;
		const origH2 = H * dsH;
		const origC2 = C * dsC;
		const depthExtent = origC2 > 1 ? origC2 : 0;
		const voxW = dsW;
		const voxH = dsH;
		const voxD = origC2 > 1 ? dsC : 1;

		// Convert slice ranges to downsampled coords
		const cMin = Math.floor(_sliceC[0] / dsC);
		const cMax = Math.min(C - 1, Math.floor(_sliceC[1] / dsC));
		const hMin = Math.floor(_sliceH[0] / dsH);
		const hMax = Math.min(H - 1, Math.floor(_sliceH[1] / dsH));
		const wMin = Math.floor(_sliceW[0] / dsW);
		const wMax = Math.min(W - 1, Math.floor(_sliceW[1] / dsW));

		renderer.setUniforms(
			_threshold, _opacity,
			[cMin, cMax], [hMin, hMax], [wMin, wMax],
			C, H, W,
			voxW, voxH, voxD,
			origW2 / 2, origH2 / 2, depthExtent / 2,
			minVal, maxVal,
			_alphaPower,
		);
	});

	// Update colormap when scheme changes
	$effect(() => {
		if (!renderer) return;
		const stops = COLOR_SCHEMES[colorScheme] ?? COLOR_SCHEMES.plasma;
		renderer.setColormap(stops);
	});

	// Update hovered voxel highlight
	$effect(() => {
		if (!renderer) return;
		const hv = hoveredVoxel;
		if (!hv) {
			renderer.clearHoveredVoxel();
			return;
		}
		const { C, H, W, dsC, dsH, dsW } = volumeData;
		const origC2 = C * dsC;
		const depthExtent = origC2 > 1 ? origC2 : 0;
		const voxW = dsW;
		const voxH = dsH;
		const voxD = origC2 > 1 ? dsC : 1;
		const origW2 = W * dsW;
		const origH2 = H * dsH;

		const ox = hv.xi * voxW - origW2 / 2;
		const oy = hv.yi * voxH - origH2 / 2;
		const oz = hv.ci * voxD - depthExtent / 2;
		renderer.setHoveredVoxel(ox, oy, oz, voxW, voxH, voxD);
	});

	// --- 2D overlay: axes, info text, colorbar ---
	$effect(() => {
		if (!overlayCanvas) return;
		const { C, H, W, minVal, maxVal, dsC, dsH, dsW } = volumeData;
		const _rotX = rotX;
		const _rotY = rotY;
		const _cw = containerW;
		const _ch = containerH;
		const _userZoom = userZoom;
		const _pivotX = pivotX;
		const _pivotY = pivotY;
		const _pivotZ = pivotZ;
		const _colorScheme = colorScheme;

		const cw = Math.max(1, Math.floor(_cw));
		const ch = Math.max(1, Math.floor(_ch));
		overlayCanvas.width = cw;
		overlayCanvas.height = ch;

		const ctx = overlayCanvas.getContext('2d');
		if (!ctx) return;
		ctx.clearRect(0, 0, cw, ch);

		if (C === 0 || H === 0 || W === 0) return;

		const cosX = Math.cos(_rotX * Math.PI / 180);
		const sinX = Math.sin(_rotX * Math.PI / 180);
		const cosY = Math.cos(_rotY * Math.PI / 180);
		const sinY = Math.sin(_rotY * Math.PI / 180);

		const origW2 = W * dsW;
		const origH2 = H * dsH;
		const origC2 = C * dsC;
		const depthExtent = origC2 > 1 ? origC2 : 0;

		const diagonal = Math.sqrt(origW2 * origW2 + origH2 * origH2 + depthExtent * depthExtent) || 1;
		const margin = 80;
		const baseScale = Math.min((cw - margin) / diagonal, (ch - margin) / diagonal);
		const scale = baseScale * _userZoom;

		const rpx = _pivotX * cosY - _pivotZ * sinY;
		const rpz = _pivotX * sinY + _pivotZ * cosY;
		const rpy = _pivotY * cosX - rpz * sinX;
		const centerX = cw / 2 - rpx * scale;
		const centerY = ch / 2 - rpy * scale;

		function volToScreen(vx: number, vy: number, vz: number): [number, number] {
			return project3D(vx, vy, vz, cosX, sinX, cosY, sinY, centerX, centerY, scale);
		}

		// --- Draw axes along volume edges ---
		const halfW = origW2 / 2;
		const halfH = origH2 / 2;
		const halfD = depthExtent / 2;

		const axOrigin: [number, number, number] = [-halfW, -halfH, -halfD];

		const edgeAxes: { from: [number, number, number]; to: [number, number, number]; label: string; color: string; dimSize: number }[] = [
			{ from: axOrigin, to: [halfW, -halfH, -halfD], label: 'W', color: '#ef4444', dimSize: origW2 },
			{ from: axOrigin, to: [-halfW, halfH, -halfD], label: 'H', color: '#22c55e', dimSize: origH2 },
			{ from: axOrigin, to: [-halfW, -halfH, halfD], label: 'C', color: '#3b82f6', dimSize: origC2 },
		];

		for (const axis of edgeAxes) {
			if (axis.label === 'C' && C <= 1) continue;

			const [p1x, p1y] = volToScreen(...axis.from);
			const [p2x, p2y] = volToScreen(...axis.to);

			ctx.strokeStyle = axis.color;
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			ctx.moveTo(p1x, p1y);
			ctx.lineTo(p2x, p2y);
			ctx.stroke();

			const numTicks = Math.min(10, axis.dimSize);
			const tickLen = 5;
			const dx = p2x - p1x;
			const dy = p2y - p1y;
			const len = Math.sqrt(dx * dx + dy * dy) || 1;
			const perpX = -dy / len;
			const perpY = dx / len;

			for (let ti = 0; ti <= numTicks; ti++) {
				const t = ti / numTicks;
				const tx2 = p1x + dx * t;
				const ty2 = p1y + dy * t;
				ctx.strokeStyle = axis.color;
				ctx.lineWidth = 1;
				ctx.beginPath();
				ctx.moveTo(tx2, ty2);
				ctx.lineTo(tx2 + perpX * tickLen, ty2 + perpY * tickLen);
				ctx.stroke();
			}

			ctx.fillStyle = axis.color;
			const labelX = p2x + (p2x - p1x) * 0.06 + perpX * 12;
			const labelY = p2y + (p2y - p1y) * 0.06 + perpY * 12;
			ctx.font = '11px monospace';
			ctx.fillText(`${axis.label}: ${axis.dimSize}`, labelX, labelY);
		}

		// Draw info text and colorbar
		ctx.fillStyle = '#9ca3af';
		ctx.font = '12px monospace';
		ctx.textAlign = 'left';
		ctx.textBaseline = 'top';
		ctx.fillText(`Volume: ${origC2} x ${origH2} x ${origW2}${dsC > 1 ? ` (ds ${dsC}x)` : ''}`, 8, 8);
		ctx.fillText(`Diff range: ${formatValue(minVal)} - ${formatValue(maxVal)}`, 8, 24);

		// Mini colorbar
		const barX = cw - 170;
		const barY = 10;
		const barW = 120;
		const barH = 12;
		for (let px = 0; px < barW; px++) {
			const t = px / (barW - 1);
			const [r, g, b] = sampleColormap(t);
			ctx.fillStyle = `rgb(${r},${g},${b})`;
			ctx.fillRect(barX + px, barY, 1, barH);
		}
		ctx.strokeStyle = '#666';
		ctx.lineWidth = 1;
		ctx.strokeRect(barX, barY, barW, barH);
		ctx.fillStyle = '#9ca3af';
		ctx.font = '10px monospace';
		ctx.textAlign = 'left';
		ctx.fillText(formatValue(minVal), barX, barY + barH + 3);
		ctx.textAlign = 'right';
		ctx.fillText(formatValue(maxVal), barX + barW, barY + barH + 3);
	});
</script>

<div class="h-full flex flex-col gap-2">
	<div class="flex flex-wrap items-center gap-4 text-xs text-gray-400 shrink-0 w-full">
		<label class="flex items-center gap-1.5 flex-1 min-w-[10rem]">
			<span class="whitespace-nowrap shrink-0">Hide errors below</span>
			<input
				use:rangeScroll
				type="range"
				min="0"
				max="1"
				step="0.01"
				bind:value={threshold}
				class="flex-1 accent-purple-500"
			/>
			<span class="w-8 text-right font-mono text-gray-300 shrink-0">{threshold.toFixed(2)}</span>
		</label>
		<label class="flex items-center gap-1.5 flex-1 min-w-[10rem]">
			<span class="whitespace-nowrap shrink-0">Voxel opacity</span>
			<input
				use:rangeScroll
				type="range"
				min="0.05"
				max="1"
				step="0.01"
				bind:value={opacitySlider}
				class="flex-1 accent-purple-500"
			/>
			<span class="w-8 text-right font-mono text-gray-300 shrink-0">{opacity.toFixed(2)}</span>
		</label>
		<label class="flex items-center gap-1.5 flex-1 min-w-[10rem]">
			<span class="whitespace-nowrap shrink-0">Alpha curve</span>
			<input
				use:rangeScroll
				type="range"
				min="0.5"
				max="5"
				step="0.25"
				bind:value={alphaPower}
				class="flex-1 accent-purple-500"
			/>
			<span class="w-8 text-right font-mono text-gray-300 shrink-0">{alphaPower.toFixed(1)}</span>
		</label>
		<label class="flex items-center gap-1.5">
			<span class="whitespace-nowrap">Colors</span>
			<select use:rangeScroll bind:value={colorScheme} class="bg-gray-700 text-gray-200 text-xs rounded px-1.5 py-0.5 border border-gray-600">
				{#each Object.keys(COLOR_SCHEMES) as name}
					<option value={name}>{name}</option>
				{/each}
			</select>
		</label>
		<label class="flex items-center gap-1.5 flex-1 min-w-[10rem]">
			<span class="whitespace-nowrap shrink-0">Chunk</span>
			<input
				use:rangeScroll
				type="range"
				min="1"
				max="16"
				step="1"
				bind:value={chunkSize}
				class="flex-1 accent-purple-500"
			/>
			<span class="w-8 text-right font-mono text-gray-300 shrink-0">{chunkSize}x</span>
		</label>
		<span class="text-gray-500 italic">Drag to orbit · Shift-drag or middle-drag to pan · Scroll to zoom</span>
	</div>
	<!-- Axis slice range sliders -->
	{#if volumeData}
		<div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-400 shrink-0 w-full">
			{#each [
				{ label: 'C', color: 'blue', max: volumeData.origC - 1, slice: sliceC, set: (v: [number, number]) => sliceC = v },
				{ label: 'H', color: 'green', max: volumeData.origH - 1, slice: sliceH, set: (v: [number, number]) => sliceH = v },
				{ label: 'W', color: 'red', max: volumeData.origW - 1, slice: sliceW, set: (v: [number, number]) => sliceW = v },
			] as axis}
				{#if axis.max > 0}
					<div class="flex items-center gap-1.5 flex-1 min-w-[10rem]">
						<span class="w-3 font-semibold shrink-0" style="color: {axis.color === 'blue' ? '#3b82f6' : axis.color === 'green' ? '#22c55e' : '#ef4444'}">{axis.label}</span>
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<div
							class="relative flex-1 h-5"
							ondblclick={() => axis.set([0, axis.max])}
						>
							<!-- Track background -->
							<div class="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-1 rounded bg-gray-700"></div>
							<!-- Active range bar -->
							<div
								class="absolute top-1/2 -translate-y-1/2 h-1 rounded"
								style="left: {axis.max > 0 ? (axis.slice[0] / axis.max) * 100 : 0}%; right: {axis.max > 0 ? (1 - axis.slice[1] / axis.max) * 100 : 0}%; background: {axis.color === 'blue' ? '#3b82f6' : axis.color === 'green' ? '#22c55e' : '#ef4444'};"
							></div>
							<!-- Min slider -->
							<input
								use:rangeScroll
								type="range"
								min="0"
								max={axis.max}
								step="1"
								value={axis.slice[0]}
								oninput={(e: Event) => {
									const el = e.target as HTMLInputElement;
									const v = parseInt(el.value);
									const clamped = Math.min(v, axis.slice[1] - 1);
									if (v !== clamped) el.value = String(clamped);
									axis.set([clamped, axis.slice[1]]);
								}}
								class="dual-range-slider absolute inset-0 w-full"
								style="--thumb-color: {axis.color === 'blue' ? '#3b82f6' : axis.color === 'green' ? '#22c55e' : '#ef4444'};"
							/>
							<!-- Max slider -->
							<input
								use:rangeScroll
								type="range"
								min="0"
								max={axis.max}
								step="1"
								value={axis.slice[1]}
								oninput={(e: Event) => {
									const el = e.target as HTMLInputElement;
									const v = parseInt(el.value);
									const clamped = Math.max(v, axis.slice[0] + 1);
									if (v !== clamped) el.value = String(clamped);
									axis.set([axis.slice[0], clamped]);
								}}
								class="dual-range-slider absolute inset-0 w-full"
								style="--thumb-color: {axis.color === 'blue' ? '#3b82f6' : axis.color === 'green' ? '#22c55e' : '#ef4444'};"
							/>
						</div>
						<span class="font-mono text-gray-300 w-20 text-right text-[10px]">{axis.slice[0]}&ndash;{axis.slice[1]} / {axis.max + 1}</span>
					</div>
				{/if}
			{/each}
		</div>
	{/if}
	<div bind:this={container} class="flex-1 min-h-0 relative">
		<canvas
			bind:this={gpuCanvas}
			class="absolute inset-0 w-full h-full rounded border border-gray-700 cursor-grab"
			class:cursor-grabbing={dragging}
			onmousedown={onMouseDown}
			onmousemove={onMouseMove}
			onmouseup={onMouseUp}
			onmouseleave={onMouseLeave}
			onwheel={onWheel}
			onauxclick={(e: MouseEvent) => e.preventDefault()}
		></canvas>
		<canvas
			bind:this={overlayCanvas}
			class="absolute inset-0 w-full h-full pointer-events-none"
		></canvas>
		{#if tooltip}
			<div
				class="fixed z-50 pointer-events-none px-3 py-2 rounded-lg border border-gray-600 bg-gray-800/95 text-xs font-mono text-gray-200 shadow-xl"
				style="left: {tooltip.x + 14}px; top: {tooltip.y + 14}px;"
			>
				<div class="text-purple-300 font-semibold mb-1">[C={tooltip.ci}, H={tooltip.iy}, W={tooltip.ix}]</div>
				<div><span class="text-gray-400">{mainLabel}:</span> {formatValue(tooltip.mainVal)}</div>
				<div><span class="text-gray-400">{refLabel}:</span> {formatValue(tooltip.refVal)}</div>
				<div><span class="text-yellow-400">Diff:</span> <span class="text-yellow-300">{formatValue(tooltip.diffVal)}</span></div>
			</div>
		{/if}
	</div>
</div>

<style>
	.dual-range-slider {
		-webkit-appearance: none;
		appearance: none;
		background: transparent;
		pointer-events: none;
		height: 100%;
		margin: 0;
	}
	.dual-range-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--thumb-color, #a855f7);
		cursor: pointer;
		pointer-events: auto;
		border: 1.5px solid #1f2937;
		box-shadow: 0 0 2px rgba(0,0,0,0.5);
	}
	.dual-range-slider::-moz-range-thumb {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--thumb-color, #a855f7);
		cursor: pointer;
		pointer-events: auto;
		border: 1.5px solid #1f2937;
		box-shadow: 0 0 2px rgba(0,0,0,0.5);
	}
	.dual-range-slider::-webkit-slider-runnable-track {
		background: transparent;
		height: 4px;
	}
	.dual-range-slider::-moz-range-track {
		background: transparent;
		height: 4px;
	}
</style>
