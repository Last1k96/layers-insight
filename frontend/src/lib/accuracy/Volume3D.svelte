<script lang="ts">
	import { getSpatialDims, formatValue } from './tensorUtils';

	let { main, ref, shape, mainLabel = 'Main', refLabel = 'Reference' }: {
		main: Float32Array;
		ref: Float32Array;
		shape: number[];
		mainLabel?: string;
		refLabel?: string;
	} = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	let container: HTMLDivElement | undefined = $state();
	let containerW = $state(800);
	let containerH = $state(600);
	let rotX = $state(-25);
	let rotY = $state(30);
	let threshold = $state(0.1);
	let opacity = $state(0.3);

	let dragging = $state(false);
	let dragStartX = $state(0);
	let dragStartY = $state(0);
	let rotStartX = $state(0);
	let rotStartY = $state(0);

	let userZoom = $state(1.0);

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

		// Downsample if too many voxels
		const total = C * H * W;
		if (total <= MAX_VOXELS) {
			return { diff, C, H, W, minVal, maxVal, dsC: 1, dsH: 1, dsW: 1, dsOrigin: null, origC: C, origH: H, origW: W };
		}

		const factor = Math.ceil(Math.cbrt(total / MAX_VOXELS));
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
		dragging = true;
		dragStartX = e.clientX;
		dragStartY = e.clientY;
		rotStartX = rotX;
		rotStartY = rotY;
		hoveredVoxel = null;
		tooltip = null;
	}

	function onMouseMove(e: MouseEvent) {
		if (dragging) {
			rotY = rotStartY + (e.clientX - dragStartX) * 0.5;
			rotX = rotStartX + (e.clientY - dragStartY) * 0.5;
			rotX = Math.max(-89, Math.min(89, rotX));
			hoveredVoxel = null;
			tooltip = null;
			return;
		}

		// Hit-test for tooltip — project each voxel center, find closest to mouse
		if (!canvas || !renderParams) return;
		const rect = canvas.getBoundingClientRect();
		const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
		const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);

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

		for (let ci = cStart; ci !== cEnd; ci += cStep) {
			for (let yi = hStart; yi !== hEnd; yi += hStep) {
				for (let xi = wStart; xi !== wEnd; xi += wStep) {
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
	}

	function onMouseLeave() {
		dragging = false;
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

	$effect(() => {
		if (!canvas) return;
		const { diff, C, H, W, minVal, maxVal } = volumeData;
		const _rotX = rotX;
		const _rotY = rotY;
		const _threshold = threshold;
		const _opacity = opacity;
		const _colorScheme = colorScheme;
		const _hoveredVoxel = hoveredVoxel;
		const _cw = containerW;
		const _ch = containerH;

		const cw = Math.max(1, Math.floor(_cw));
		const ch = Math.max(1, Math.floor(_ch));
		canvas.width = cw;
		canvas.height = ch;

		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		// Clear with dark background
		ctx.fillStyle = '#111827';
		ctx.fillRect(0, 0, cw, ch);

		if (C === 0 || H === 0 || W === 0) return;

		const span = maxVal - minVal || 1;
		const invSpan = 1 / span;

		// Projection parameters
		const cosX = Math.cos(_rotX * Math.PI / 180);
		const sinX = Math.sin(_rotX * Math.PI / 180);
		const cosY = Math.cos(_rotY * Math.PI / 180);
		const sinY = Math.sin(_rotY * Math.PI / 180);

		const { dsC, dsH, dsW } = volumeData;

		// World-space extents proportional to original tensor dimensions
		const origW = W * dsW;
		const origH = H * dsH;
		const origC = C * dsC;
		const depthExtent = origC > 1 ? origC : 0;

		// Rotation-independent base scale (worst-case diagonal)
		const diagonal = Math.sqrt(origW * origW + origH * origH + depthExtent * depthExtent) || 1;
		const margin = 80;
		const _userZoom = userZoom;
		const baseScale = Math.min((cw - margin) / diagonal, (ch - margin) / diagonal);
		const scale = baseScale * _userZoom;

		// Volume is centered at origin, so centroid is always (0,0)
		const centerX = cw / 2;
		const centerY = ch / 2;

		// Store render params for mousemove hit-testing
		renderParams = { scale, cosX, sinX, cosY, sinY, centerX, centerY, C, H, W, depthExtent, dsC, dsH, dsW };

		// --- Voxel rendering ---
		// Each voxel is 1 original-unit in each axis (dsW × dsH × dsC in downsampled grid)
		const voxW = dsW;
		const voxH = dsH;
		const voxD = origC > 1 ? dsC : 1;

		// Determine which 3 faces are visible based on camera direction
		// Camera looks from direction (sinY, sinX, cosX*cosY) approximately
		const showPosX = sinY > 0;   // right face vs left face
		const showPosY = sinX > 0;   // bottom face vs top face
		const showPosZ = cosX * cosY > 0; // front face vs back face

		// Face definitions: each face is 4 corner offsets from voxel origin (0,0,0) to (voxW, voxH, voxD)
		// We define the 3 visible faces with brightness multipliers for pseudo-lighting
		type Face = { corners: [number,number,number][]; brightness: number };
		const faces: Face[] = [];

		// X-facing face (side)
		const fx = showPosX ? voxW : 0;
		faces.push({ corners: [[fx,0,0],[fx,voxH,0],[fx,voxH,voxD],[fx,0,voxD]], brightness: 0.8 });

		// Y-facing face (top/bottom)
		const fy = showPosY ? voxH : 0;
		faces.push({ corners: [[0,fy,0],[voxW,fy,0],[voxW,fy,voxD],[0,fy,voxD]], brightness: showPosY ? 0.6 : 1.0 });

		// Z-facing face (front/back)
		const fz = showPosZ ? voxD : 0;
		faces.push({ corners: [[0,0,fz],[voxW,0,fz],[voxW,voxH,fz],[0,0+voxH,fz]], brightness: 0.9 });

		// Determine traversal order: back-to-front for painter's algorithm
		// Depth after rotation: x*sinY + z*cosY (from project3D Y-rotation)
		// C maps to Z-axis → sort by cosY; W maps to X-axis → sort by sinY; H maps to Y-axis → sort by sinX
		const cStart = cosY > 0 ? 0 : C - 1, cEnd = cosY > 0 ? C : -1, cStep = cosY > 0 ? 1 : -1;
		const hStart = sinX > 0 ? 0 : H - 1, hEnd = sinX > 0 ? H : -1, hStep = sinX > 0 ? 1 : -1;
		const wStart = sinY > 0 ? W - 1 : 0, wEnd = sinY > 0 ? -1 : W, wStep = sinY > 0 ? -1 : 1;

		// Render voxels
		for (let ci = cStart; ci !== cEnd; ci += cStep) {
			for (let yi = hStart; yi !== hEnd; yi += hStep) {
				for (let xi = wStart; xi !== wEnd; xi += wStep) {
					const v = diff[ci * H * W + yi * W + xi];
					const norm = (v - minVal) * invSpan;
					if (norm < _threshold) continue;

					// World-space origin of this voxel
					const ox = xi * voxW - origW / 2;
					const oy = yi * voxH - origH / 2;
					const oz = ci * voxD - depthExtent / 2;

					const [r, g, b] = sampleColormap(norm);
					const alpha = _opacity * (0.3 + 0.7 * norm);

					// Draw 3 visible faces
					for (const face of faces) {
						const fr = Math.round(r * face.brightness);
						const fg = Math.round(g * face.brightness);
						const fb = Math.round(b * face.brightness);

						ctx.fillStyle = `rgba(${fr},${fg},${fb},${alpha})`;
						ctx.beginPath();
						const [p0x, p0y] = project3D(ox + face.corners[0][0], oy + face.corners[0][1], oz + face.corners[0][2], cosX, sinX, cosY, sinY, centerX, centerY, scale);
						ctx.moveTo(p0x, p0y);
						for (let fi = 1; fi < 4; fi++) {
							const [px, py] = project3D(ox + face.corners[fi][0], oy + face.corners[fi][1], oz + face.corners[fi][2], cosX, sinX, cosY, sinY, centerX, centerY, scale);
							ctx.lineTo(px, py);
						}
						ctx.closePath();
						ctx.fill();
					}
				}
			}
		}

		// --- Draw hovered voxel outline ---
		if (_hoveredVoxel) {
			const hv = _hoveredVoxel;
			const ox = hv.xi * voxW - origW / 2;
			const oy = hv.yi * voxH - origH / 2;
			const oz = hv.ci * voxD - depthExtent / 2;

			// Project all 8 corners
			const c000 = project3D(ox, oy, oz, cosX, sinX, cosY, sinY, centerX, centerY, scale);
			const c100 = project3D(ox + voxW, oy, oz, cosX, sinX, cosY, sinY, centerX, centerY, scale);
			const c010 = project3D(ox, oy + voxH, oz, cosX, sinX, cosY, sinY, centerX, centerY, scale);
			const c110 = project3D(ox + voxW, oy + voxH, oz, cosX, sinX, cosY, sinY, centerX, centerY, scale);
			const c001 = project3D(ox, oy, oz + voxD, cosX, sinX, cosY, sinY, centerX, centerY, scale);
			const c101 = project3D(ox + voxW, oy, oz + voxD, cosX, sinX, cosY, sinY, centerX, centerY, scale);
			const c011 = project3D(ox, oy + voxH, oz + voxD, cosX, sinX, cosY, sinY, centerX, centerY, scale);
			const c111 = project3D(ox + voxW, oy + voxH, oz + voxD, cosX, sinX, cosY, sinY, centerX, centerY, scale);

			// Draw all 12 edges of the cube
			const edges: [[number,number],[number,number]][] = [
				[c000,c100],[c010,c110],[c001,c101],[c011,c111], // X edges
				[c000,c010],[c100,c110],[c001,c011],[c101,c111], // Y edges
				[c000,c001],[c100,c101],[c010,c011],[c110,c111], // Z edges
			];

			ctx.strokeStyle = '#ffffff';
			ctx.lineWidth = 2;
			for (const [a, b] of edges) {
				ctx.beginPath();
				ctx.moveTo(a[0], a[1]);
				ctx.lineTo(b[0], b[1]);
				ctx.stroke();
			}
		}

		// --- Draw axes along volume edges ---
		// Use project3D for consistent coordinate system (world centered at origin)
		const halfW = origW / 2;
		const halfH = origH / 2;
		const halfD = depthExtent / 2;

		function volToScreen(vx: number, vy: number, vz: number): [number, number] {
			return project3D(vx, vy, vz, cosX, sinX, cosY, sinY, centerX, centerY, scale);
		}

		// Fixed axes at the (-halfW, -halfH, -halfD) corner — origin of the volume
		const axOrigin: [number, number, number] = [-halfW, -halfH, -halfD];

		const edgeAxes: { from: [number, number, number]; to: [number, number, number]; label: string; color: string; dimSize: number }[] = [
			{ from: axOrigin, to: [halfW, -halfH, -halfD], label: 'W', color: '#ef4444', dimSize: origW },
			{ from: axOrigin, to: [-halfW, halfH, -halfD], label: 'H', color: '#22c55e', dimSize: origH },
			{ from: axOrigin, to: [-halfW, -halfH, halfD], label: 'C', color: '#3b82f6', dimSize: origC },
		];

		for (const axis of edgeAxes) {
			if (axis.label === 'C' && C <= 1) continue;

			const [p1x, p1y] = volToScreen(...axis.from);
			const [p2x, p2y] = volToScreen(...axis.to);

			// Draw axis line
			ctx.strokeStyle = axis.color;
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			ctx.moveTo(p1x, p1y);
			ctx.lineTo(p2x, p2y);
			ctx.stroke();

			// Tick marks (cap at 10)
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

			// Label dim size at the far end
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
		ctx.fillText(`Volume: ${origC} x ${origH} x ${origW}${dsC > 1 ? ` (ds ${dsC}x)` : ''}`, 8, 8);
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
	<div class="flex flex-wrap items-center gap-4 text-xs text-gray-400 shrink-0">
		<label class="flex items-center gap-1.5">
			<span class="whitespace-nowrap">Hide errors below</span>
			<input
				type="range"
				min="0"
				max="1"
				step="0.01"
				bind:value={threshold}
				class="w-24 accent-purple-500"
			/>
			<span class="w-8 text-right font-mono text-gray-300">{threshold.toFixed(2)}</span>
		</label>
		<label class="flex items-center gap-1.5">
			<span class="whitespace-nowrap">Voxel opacity</span>
			<input
				type="range"
				min="0.05"
				max="1"
				step="0.05"
				bind:value={opacity}
				class="w-24 accent-purple-500"
			/>
			<span class="w-8 text-right font-mono text-gray-300">{opacity.toFixed(2)}</span>
		</label>
		<label class="flex items-center gap-1.5">
			<span class="whitespace-nowrap">Colors</span>
			<select bind:value={colorScheme} class="bg-gray-700 text-gray-200 text-xs rounded px-1.5 py-0.5 border border-gray-600">
				{#each Object.keys(COLOR_SCHEMES) as name}
					<option value={name}>{name}</option>
				{/each}
			</select>
		</label>
		<span class="text-gray-500 italic">Drag to orbit · Scroll to zoom</span>
	</div>
	<div bind:this={container} class="flex-1 min-h-0 relative">
		<canvas
			bind:this={canvas}
			class="w-full h-full rounded border border-gray-700 cursor-grab"
			class:cursor-grabbing={dragging}
			onmousedown={onMouseDown}
			onmousemove={onMouseMove}
			onmouseup={onMouseUp}
			onmouseleave={onMouseLeave}
			onwheel={onWheel}
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
